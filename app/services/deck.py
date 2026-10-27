"""Rating Deck: an endless queue of well-known titles to rate, refilled from TMDB."""

import asyncio
import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from app.config import settings
from app.database import get_db
from app.services.catalog import cache_tmdb_titles, format_title_row
from app.services.explain import personalized_reason
from app.services.ranking import build_taste_signal, rank_candidates
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
_refill_task = None  # held so the running refill is not garbage-collected

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

    global _refill_task
    _refill_task = asyncio.get_running_loop().create_task(runner())


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
    """The unseen quality title the For You ranker scores highest, or None without positive evidence."""
    signal = build_taste_signal()
    if signal.ratings_count < 2 or not signal.liked_count:
        return None
    with get_db() as conn:
        ineligible = {r[0] for r in conn.execute(f"SELECT id FROM titles EXCEPT SELECT t.id {UNSEEN_QUALITY}",
                                                 QUALITY_FLOOR)}
        ranked = rank_candidates({"_excluded_title_ids": ineligible}, signal=signal)
        if not ranked or ranked[0]["raw_similarity"] <= 0:
            return None
        best = ranked[0]
        row = conn.execute("SELECT * FROM titles WHERE id = ?", (best["title_id"],)).fetchone()
    reason = personalized_reason(best, signal.liked_genres, signal.liked_creators, signal.loved_titles)
    return {**format_title_row(row), "is_suggestion": True, "suggestion_reason": reason}
