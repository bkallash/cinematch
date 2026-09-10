import json
import logging
from pathlib import Path

from app.database import get_db
from app.services.embeddings import embedding_service

logger = logging.getLogger(__name__)


async def seed_starter_catalog():
    """Load bundled metadata, then batch missing or incompatible embeddings."""
    catalog_path = Path(__file__).resolve().parent.parent / "data" / "starter_catalog.json"
    if not catalog_path.exists():
        logger.warning("Starter catalog not found at %s", catalog_path)
        return
    titles = json.loads(catalog_path.read_text(encoding="utf-8"))
    with get_db() as conn:
        for item in titles:
            conn.execute("""
                INSERT INTO titles (
                    tmdb_id, media_type, title, original_title, release_year, overview,
                    poster_path, backdrop_path, genres, director_or_creator, cast_top,
                    vote_average, vote_count, popularity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tmdb_id, media_type) DO UPDATE SET poster_path = excluded.poster_path
            """, (
                item["tmdb_id"], item["media_type"], item["title"], item.get("original_title"),
                item.get("release_year"), item.get("overview"), item.get("poster_path"),
                item.get("backdrop_path"), json.dumps(item.get("genres") or []),
                item.get("director_or_creator"), json.dumps(item.get("cast_top") or []),
                item.get("vote_average", 0.0), item.get("vote_count", 0), item.get("popularity", 0.0),
            ))
        # Include dynamically discovered titles when changing embedding models too.
        pending = conn.execute("""
            SELECT * FROM titles
            WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?
        """, (embedding_service.model_key,)).fetchall()
        if pending:
            conn.execute("DELETE FROM for_you_cache")

    for start in range(0, len(pending), 64):
        batch = pending[start:start + 64]
        texts = []
        for row in batch:
            item = dict(row)
            item["genres"] = json.loads(item["genres"] or "[]")
            item["cast_top"] = json.loads(item["cast_top"] or "[]")
            texts.append(embedding_service.build_title_embedding_text(item))
        try:
            vectors = await embedding_service.get_embeddings(texts)
        except Exception:
            # Preserve catalog metadata. The next startup retries the missing vectors.
            logger.exception("Embedding batch failed; remaining titles will be retried on startup")
            break
        with get_db() as conn:
            conn.executemany("""
                UPDATE titles SET embedding = ?, embedding_dim = ?, embedding_model = ? WHERE id = ?
            """, [(embedding_service.vec_to_bytes(vec), len(vec), embedding_service.model_key, row["id"])
                  for row, vec in zip(batch, vectors)])
    logger.info("Starter catalog ready; checked %d titles for embedding updates", len(pending))
