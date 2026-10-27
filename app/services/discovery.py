"""Grow the local catalog from TMDB when local candidates are too few or too weak."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from app.database import get_db
from app.services.catalog import cache_tmdb_titles
from app.services.embeddings import embedding_service
from app.services.ranking import TasteSignal, matches_intent, media_evidence, normalize_intent
from app.services.tmdb import GENRE_NAME_TO_ID, tmdb_service

logger = logging.getLogger(__name__)


async def try_discovery(intent: Dict[str, Any]) -> List[int]:
    try:
        return await discover_and_cache(intent)
    except Exception:
        logger.exception("Discovery unavailable; retaining eligible local Titles")
        return []


def for_you_discovery_intent(signal: TasteSignal, media_filter: Optional[str]) -> Dict[str, Any]:
    genres_by_media = {}
    for media in ("movie", "tv"):
        affinities: Dict[str, float] = {}
        for entry in media_evidence(signal.rating_rows, media):
            if entry["weight"] > 0:
                for genre in json.loads(entry["row"]["genres"] or "[]"):
                    affinities[genre] = affinities.get(genre, 0) + entry["weight"]
        genres_by_media[media] = sorted(affinities, key=lambda g: (-affinities[g], g))[:1]
    return {"media_type": media_filter, "genres": genres_by_media.get(media_filter or "", []),
            "_genres_by_media": genres_by_media}


async def discover_and_cache(intent: Dict[str, Any]) -> List[int]:
    """Fetch candidates from TMDB discover, compute embeddings, and insert into SQLite concurrently."""
    if not intent.get("media_type"):
        results = await asyncio.gather(*(
            discover_and_cache(normalize_intent({
                **intent, "media_type": media,
                "genres": intent.get("_genres_by_media", {}).get(media, intent.get("genres", [])),
            }))
            for media in ("movie", "tv")
        ))
        return list(dict.fromkeys(tid for group in results for tid in group))
    media_type = intent.get("media_type") or "movie"
    genres = intent.get("genres") or []
    year_min = intent.get("year_min")
    year_max = intent.get("year_max")

    if intent.get("person"):
        tmdb_items = await tmdb_service.find_person_titles(intent["person"], media_type)
    else:
        # Use the primary inferred genre to broaden discovery. It is never a
        # mandatory local filter or an intersection of every reference genre.
        discovery_genres = genres or intent.get("similarity_genres", [])[:1]
        genre_ids = [str(GENRE_NAME_TO_ID[g.casefold()]) for g in discovery_genres if g.casefold() in GENRE_NAME_TO_ID]
        tmdb_items = await tmdb_service.discover_titles(
            media_type=media_type, with_genres=",".join(genre_ids) or None,
            year_min=year_min, year_max=year_max, limit=20,
        )

    if not tmdb_items:
        return []

    def prepare(item, details):
        full = dict(details or item)
        if intent.get("person"):
            # Details may omit the requested person; keep the credit discovery found.
            full["cast_top"] = list(dict.fromkeys([*(full.get("cast_top") or []), *(item.get("cast_top") or [])]))
            full["director_or_creator"] = full.get("director_or_creator") or item.get("director_or_creator")
        return full if matches_intent(full, intent) else None

    with get_db() as conn:
        existing = {(r["tmdb_id"], r["media_type"]) for r in conn.execute(
            "SELECT tmdb_id, media_type, genres, release_year, cast_top, director_or_creator FROM titles WHERE embedding_model = ? AND embedding IS NOT NULL",
            (embedding_service.model_key,),
        ) if matches_intent(dict(r), intent)}
    fresh = [item for item in tmdb_items if (item["tmdb_id"], item["media_type"]) not in existing
             and matches_intent(item, intent)]
    return await cache_tmdb_titles(fresh[:8], prepare, update=(
        "embedding", "embedding_dim", "embedding_model", "director_or_creator", "cast_top", "genres", "overview"))

