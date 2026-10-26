"""Local title catalog: row formatting, search, and caching TMDB titles with embeddings."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from app.config import settings
from app.database import get_db, upsert_title
from app.services.embeddings import embedding_service
from app.services.tmdb import tmdb_service

logger = logging.getLogger(__name__)

# Titles joined with the user's rating and watchlist state.
TITLE_WITH_USER_STATE = """
    SELECT t.*, r.score AS user_rating, r.aspect_tags AS user_aspect_tags, r.notes AS user_notes,
           w.id IS NOT NULL AS is_in_watchlist
    FROM titles t
    LEFT JOIN ratings r ON t.id = r.title_id
    LEFT JOIN watchlist w ON t.id = w.title_id
"""
SEARCH_LIMIT = 12
TMDB_SEARCH_BELOW = 4  # fall back to live TMDB search when local results are this sparse


def format_title_row(row) -> Dict[str, Any]:
    """Turn a titles row (or title dict) into the template-ready shape with decoded JSON fields."""
    title = dict(row)
    title.pop("embedding", None)
    for key in ("genres", "cast_top", "user_aspect_tags"):
        value = title.get(key)
        title[key] = json.loads(value) if isinstance(value, str) else (value or [])
    for key in ("original_title", "backdrop_path", "director_or_creator", "user_rating",
                "user_notes", "suggestion_reason", "match_score"):
        title.setdefault(key, None)
    title["is_in_watchlist"] = bool(title.get("is_in_watchlist"))
    title["is_suggestion"] = bool(title.get("is_suggestion"))
    return title


def popular_titles(limit: int = SEARCH_LIMIT) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(f"{TITLE_WITH_USER_STATE} ORDER BY t.popularity DESC LIMIT ?", (limit,)).fetchall()
    return [format_title_row(r) for r in rows]


async def search_catalog(query: str) -> List[Dict[str, Any]]:
    """Search local titles; top up from live TMDB search when few match."""
    if not query:
        return []
    with get_db() as conn:
        rows = conn.execute(f"""{TITLE_WITH_USER_STATE}
            WHERE t.title LIKE ? OR t.original_title LIKE ?
            ORDER BY t.popularity DESC LIMIT ?
        """, (f"%{query}%", f"%{query}%", SEARCH_LIMIT)).fetchall()
    results = [format_title_row(r) for r in rows]
    if len(results) >= TMDB_SEARCH_BELOW or not settings.TMDB_API_KEY:
        return results

    seen = {(r["tmdb_id"], r["media_type"]) for r in results}
    fresh = [i for i in await tmdb_service.search_titles(query)
             if (i["tmdb_id"], i["media_type"]) not in seen][:SEARCH_LIMIT - len(results)]
    with get_db() as conn:
        for item in fresh:
            title_id = upsert_title(conn, item)
            results.append(format_title_row(
                conn.execute(f"{TITLE_WITH_USER_STATE} WHERE t.id = ?", (title_id,)).fetchone()))
    return results


async def cache_tmdb_titles(
    items: List[Dict[str, Any]],
    prepare: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]] = None,
    update: Iterable[str] = (),
) -> List[int]:
    """Fetch details and embeddings for TMDB items concurrently, then store them.

    `prepare(item, details)` may enrich or reject (return None) each title.
    `update` names the columns overwritten when the title already exists.
    Returns the database ids of stored titles.
    """
    sem = asyncio.Semaphore(4)

    async def fetch(item):
        async with sem:
            try:
                details = await tmdb_service.get_title_details(item["tmdb_id"], item["media_type"])
                full = prepare(item, details) if prepare else (details or {**item, "director_or_creator": None, "cast_top": []})
                if full is None:
                    return None
                vec = await embedding_service.get_embedding(embedding_service.build_title_embedding_text(full))
                return full, (embedding_service.vec_to_bytes(vec), len(vec), embedding_service.model_key)
            except Exception as e:
                logger.error(f"Error caching title {item.get('title')}: {e}")
                return None

    ids = []
    results = await asyncio.gather(*(fetch(item) for item in items))
    with get_db() as conn:
        for full, embedding in filter(None, results):
            try:
                ids.append(upsert_title(conn, full, embedding, update))
            except Exception as e:
                logger.error(f"Error inserting title {full.get('title')}: {e}")
    return ids


async def ensure_title_embedding(title_id: int) -> None:
    """Make sure a rated title has full details and a current embedding."""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM titles WHERE id = ?", (title_id,)).fetchone()
        if not row:
            return
        if row["embedding"] and row["director_or_creator"] and row["embedding_model"] == embedding_service.model_key:
            return

        title = format_title_row(row)
        details = {}
        if settings.TMDB_API_KEY and (not title["director_or_creator"] or not title["cast_top"]):
            details = await tmdb_service.get_title_details(title["tmdb_id"], title["media_type"]) or {}
        vec = await embedding_service.get_embedding(embedding_service.build_title_embedding_text({**title, **details}))

        with get_db() as conn:
            conn.execute("""
                UPDATE titles SET
                    overview = COALESCE(?, overview),
                    genres = COALESCE(?, genres),
                    director_or_creator = COALESCE(?, director_or_creator),
                    cast_top = COALESCE(?, cast_top),
                    imdb_id = COALESCE(?, imdb_id),
                    imdb_rating = COALESCE(?, imdb_rating),
                    embedding = ?, embedding_dim = ?, embedding_model = ?
                WHERE id = ?
            """, (
                details.get("overview"),
                json.dumps(details["genres"]) if "genres" in details else None,
                details.get("director_or_creator"),
                json.dumps(details.get("cast_top") or []) if details else None,
                details.get("imdb_id"),
                details.get("imdb_rating"),
                embedding_service.vec_to_bytes(vec), len(vec), embedding_service.model_key,
                title_id,
            ))
            # A rated title's new vector changes the taste signal like a rating edit would.
            conn.execute("DELETE FROM for_you_cache")
            conn.execute("UPDATE taste_dossiers SET is_dirty = 1")
            conn.execute("UPDATE ratings_state SET revision = revision + 1 WHERE id = 1")
    except Exception as e:
        logger.error(f"Error ensuring embedding for title #{title_id}: {e}")
