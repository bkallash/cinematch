"""Rating Deck: an endless queue of well-known titles to rate, refilled from TMDB."""

import asyncio
import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings
from app.database import get_db
from app.services.catalog import cache_tmdb_titles, format_title_row
from app.services.embeddings import embedding_service
from app.services.orchestrator import orchestrator_service
from app.services.tmdb import GENRE_NAME_TO_ID, tmdb_service

logger = logging.getLogger(__name__)

# Deck candidates need enough audience evidence to be useful for taste
# calibration. Similarity can rank eligible Titles, but never waive this floor.
DECK_MIN_VOTE_COUNT = 800
DECK_MIN_VOTE_AVERAGE = 6.4
DECK_REFILL_LOW_WATERMARK = 15  # refill in the background below this many unseen titles
DECK_SUGGESTION_CADENCE = 3     # every 3rd card may be a personalized suggestion
DECK_REFILL_PAGES_PER_ENDPOINT = 2
DECK_MAX_TMDB_PAGE = 500
DECK_REFILL_INSERT_LIMIT = 16

_deck_refill_lock = asyncio.Lock()

UNSEEN_QUALITY = """
    FROM titles t
    LEFT JOIN ratings r ON t.id = r.title_id
    LEFT JOIN skipped_titles s ON t.id = s.title_id
    LEFT JOIN watchlist w ON t.id = w.title_id
    WHERE r.id IS NULL AND s.id IS NULL AND w.id IS NULL
      AND t.vote_count >= ? AND t.vote_average >= ?
"""
QUALITY_FLOOR = (DECK_MIN_VOTE_COUNT, DECK_MIN_VOTE_AVERAGE)


def _next_unseen_title() -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(f"SELECT t.* {UNSEEN_QUALITY} ORDER BY t.popularity DESC, RANDOM() LIMIT 1",
                           QUALITY_FLOOR).fetchone()
    return format_title_row(row) if row else None


def _take_deck_cursor(pages: int) -> int:
    """Return the next TMDB page to fetch and advance the cursor, wrapping at TMDB's page cap."""
    with get_db() as conn:
        page = conn.execute("SELECT next_page FROM deck_refill_state WHERE id = 1").fetchone()[0]
        nxt = page + pages if page + pages <= DECK_MAX_TMDB_PAGE else 1
        conn.execute("UPDATE deck_refill_state SET next_page = ? WHERE id = 1", (nxt,))
    return page


def _get_and_increment_deck_serve_count() -> int:
    with get_db() as conn:
        count = conn.execute("SELECT serve_count FROM deck_refill_state WHERE id = 1").fetchone()[0]
        conn.execute("UPDATE deck_refill_state SET serve_count = serve_count + 1 WHERE id = 1")
    return count


async def refill_deck_with_preferences_and_famous() -> int:
    """Cache fresh titles from three sources: TMDB recommendations for the user's
    favorites, discovery in their most-liked genres, and famous/top-rated titles."""
    if not settings.TMDB_API_KEY:
        return 0

    with get_db() as conn:
        existing = {(r["tmdb_id"], r["media_type"]) for r in conn.execute("SELECT tmdb_id, media_type FROM titles")}
        liked = conn.execute("""
            SELECT r.score, t.tmdb_id, t.media_type, t.genres
            FROM ratings r JOIN titles t ON r.title_id = t.id
            WHERE r.score >= 4
            ORDER BY r.score DESC, r.created_at DESC
        """).fetchall()

    candidates: List[Dict[str, Any]] = []
    favorites = [r for r in liked if r["score"] >= 5][:3] or liked[:3]
    for fav in favorites:
        candidates += await tmdb_service.get_title_recommendations(fav["tmdb_id"], fav["media_type"], limit=8)

    genre_counts = Counter(g for r in liked for g in json.loads(r["genres"] or "[]"))
    for genre, _ in genre_counts.most_common(2):
        if genre_id := GENRE_NAME_TO_ID.get(genre.lower()):
            for media_type in ("movie", "tv"):
                candidates += await tmdb_service.discover_titles(
                    media_type=media_type, with_genres=str(genre_id), vote_count_gte=250,
                    vote_average_gte=6.5, limit=8)

    candidates += await tmdb_service.get_famous_titles(
        min_vote_count=DECK_MIN_VOTE_COUNT, min_vote_average=DECK_MIN_VOTE_AVERAGE,
        pages_per_endpoint=DECK_REFILL_PAGES_PER_ENDPOINT,
        start_page=_take_deck_cursor(DECK_REFILL_PAGES_PER_ENDPOINT), limit=20)

    fresh = {}
    for c in candidates:
        key = (c["tmdb_id"], c["media_type"])
        if key not in existing:
            fresh.setdefault(key, c)
    if not fresh:
        logger.warning("Deck refill: no fresh candidates found.")
        return 0

    inserted = len(await cache_tmdb_titles(list(fresh.values())[:DECK_REFILL_INSERT_LIMIT]))
    logger.info(f"Deck refill: cached {inserted} fresh titles (preferences + famous).")
    return inserted


def schedule_deck_refill_if_needed() -> None:
    """Start a background refill unless one is already running."""
    async def runner():
        if _deck_refill_lock.locked():
            return
        async with _deck_refill_lock:
            try:
                await refill_deck_with_preferences_and_famous()
            except Exception as e:
                logger.error(f"Background deck refill error: {e}")

    asyncio.get_running_loop().create_task(runner())


async def get_next_deck_title() -> Optional[Dict[str, Any]]:
    """Pick the next Deck card. The Deck never runs out:

    1. Every DECK_SUGGESTION_CADENCE-th card is a personalized suggestion when one exists.
    2. Otherwise the most popular unseen quality title.
    3. If none remain, refill from TMDB synchronously and retry.
    4. Finally, recycle the oldest skipped quality title.
    """
    with get_db() as conn:
        remaining = conn.execute(f"SELECT COUNT(*) {UNSEEN_QUALITY}", QUALITY_FLOOR).fetchone()[0]
    if remaining < DECK_REFILL_LOW_WATERMARK:
        schedule_deck_refill_if_needed()

    if _get_and_increment_deck_serve_count() % DECK_SUGGESTION_CADENCE == 0:
        try:
            if suggestion := await get_deck_suggestion():
                return suggestion
        except Exception as e:
            logger.error(f"Error fetching personalized deck suggestion: {e}")

    if title := _next_unseen_title():
        return title
    if settings.TMDB_API_KEY:
        await refill_deck_with_preferences_and_famous()
        if title := _next_unseen_title():
            return title

    # Never serve low-signal titles merely because the fresh pool ran out.
    with get_db() as conn:
        row = conn.execute("""
            SELECT t.* FROM titles t
            JOIN skipped_titles s ON t.id = s.title_id
            LEFT JOIN ratings r ON t.id = r.title_id
            LEFT JOIN watchlist w ON t.id = w.title_id
            WHERE r.id IS NULL AND w.id IS NULL
              AND t.vote_count >= ? AND t.vote_average >= ?
            ORDER BY s.created_at ASC LIMIT 1
        """, QUALITY_FLOOR).fetchone()
        if row:
            conn.execute("DELETE FROM skipped_titles WHERE title_id = ?", (row["id"],))
    return format_title_row(row) if row else None


async def get_deck_title_by_id(title_id: int) -> Optional[Dict[str, Any]]:
    """Load a specific title for the Deck (e.g. picked from Search), fetching credits on demand."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM titles WHERE id = ?", (title_id,)).fetchone()
    if not row:
        return None
    title = format_title_row(row)
    if not settings.TMDB_API_KEY or (title["director_or_creator"] and title["cast_top"]):
        return title

    details = await tmdb_service.get_title_details(title["tmdb_id"], title["media_type"])
    if details:
        title.update(director_or_creator=details.get("director_or_creator"),
                     cast_top=details.get("cast_top") or [],
                     imdb_id=details.get("imdb_id"),
                     imdb_rating=details.get("imdb_rating") or details.get("vote_average", 0.0))
        with get_db() as conn:
            conn.execute("""
                UPDATE titles SET director_or_creator = ?, cast_top = ?, imdb_id = ?, imdb_rating = ?
                WHERE id = ?
            """, (title["director_or_creator"], json.dumps(title["cast_top"]), title["imdb_id"],
                  title["imdb_rating"], title_id))
    return title


async def get_deck_suggestion() -> Optional[Dict[str, Any]]:
    """Return the unseen quality title closest to the user's taste, or None without enough signal."""
    signal = orchestrator_service.build_taste_signal()
    if signal.ratings_count < 2 or signal.taste_vector is None:
        return None

    with get_db() as conn:
        rows = conn.execute(f"SELECT t.* {UNSEEN_QUALITY} AND t.embedding IS NOT NULL AND t.embedding_model = ?",
                            (*QUALITY_FLOOR, embedding_service.model_key)).fetchall()

    best = None
    for row in rows:
        vec = embedding_service.bytes_to_vec(row["embedding"])
        if vec.shape != signal.taste_vector.shape or not np.isfinite(vec).all():
            continue
        sim = float(np.dot(vec, signal.taste_vector))
        genres = json.loads(row["genres"] or "[]")
        genre_matches = [g for g in genres if g in signal.liked_genres]
        creator_match = bool(row["director_or_creator"] and row["director_or_creator"] in signal.liked_creators)
        score = (sim + min((row["vote_average"] or 0.0) / 10.0, 1.0) * 0.05
                 + min(len(genre_matches) * 0.15, 0.3) + (0.25 if creator_match else 0.0))
        if best is None or score > best[0]:
            best = (score, sim, row, genres, genre_matches, creator_match)

    if best is None:
        return None
    _, sim, row, genres, genre_matches, creator_match = best
    if sim < 0.35 and not genre_matches and not creator_match:
        return None

    if creator_match:
        reason = f"Matches your affinity for {row['director_or_creator']}"
    elif genre_matches:
        reason = f"Matches your taste in {' & '.join(genre_matches[:2])}"
    else:
        reason = f"Tailored to your taste, offering a compelling {genres[0] if genres else 'film'} experience."
    return {**format_title_row(row), "is_suggestion": True, "suggestion_reason": reason,
            "match_score": min(98, max(75, int(sim * 100)))}
