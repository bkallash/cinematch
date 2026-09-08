import json
import logging
from pathlib import Path
from app.config import settings
from app.database import get_db
from app.services.embeddings import embedding_service
from app.services.tmdb import tmdb_service

logger = logging.getLogger(__name__)

async def seed_starter_catalog():
    """Load bundled starter catalog into SQLite and compute vector embeddings."""
    catalog_path = Path(__file__).resolve().parent.parent / "data" / "starter_catalog.json"
    if not catalog_path.exists():
        logger.warning(f"Starter catalog not found at {catalog_path}")
        return

    with open(catalog_path, "r", encoding="utf-8") as f:
        titles = json.load(f)

    logger.info(f"Checking starter catalog: {len(titles)} titles available.")
    inserted_count = 0

    for item in titles:
        # Check if title already exists
        with get_db() as conn:
            existing = conn.execute(
                "SELECT id, embedding, poster_path FROM titles WHERE tmdb_id = ? AND media_type = ?",
                (item["tmdb_id"], item["media_type"])
            ).fetchone()

        # Always sync bundled poster_path so refreshed TMDB artwork propagates
        # to existing rows (TMDB retires old image paths with 404s).
        if existing and existing["poster_path"] != item.get("poster_path"):
            with get_db() as conn:
                conn.execute(
                    "UPDATE titles SET poster_path = ? WHERE id = ?",
                    (item.get("poster_path"), existing["id"])
                )
                inserted_count += 1

        if existing and existing["embedding"] is not None:
            continue

        # Build embedding text and calculate vector
        embed_text = embedding_service.build_title_embedding_text(item)
        vec = await embedding_service.get_embedding(embed_text)
        blob = embedding_service.vec_to_bytes(vec)

        with get_db() as conn:
            if existing:
                conn.execute(
                    "UPDATE titles SET embedding = ?, embedding_dim = ? WHERE id = ?",
                    (blob, len(vec), existing["id"])
                )
            else:
                conn.execute("""
                    INSERT INTO titles (
                        tmdb_id, media_type, title, original_title, release_year,
                        overview, poster_path, backdrop_path, genres,
                        director_or_creator, cast_top, vote_average, vote_count,
                        popularity, embedding, embedding_dim
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["tmdb_id"],
                    item["media_type"],
                    item["title"],
                    item.get("original_title"),
                    item.get("release_year"),
                    item.get("overview"),
                    item.get("poster_path"),
                    item.get("backdrop_path"),
                    json.dumps(item.get("genres") or []),
                    item.get("director_or_creator"),
                    json.dumps(item.get("cast_top") or []),
                    item.get("vote_average", 0.0),
                    item.get("vote_count", 0),
                    item.get("popularity", 0.0),
                    blob,
                    len(vec)
                ))
            inserted_count += 1

    logger.info(f"Seeder complete. Processed {inserted_count} titles with embeddings.")
