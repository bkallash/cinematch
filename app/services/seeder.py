import json
import logging
from pathlib import Path

from app.database import get_db, upsert_title
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
            upsert_title(conn, item, update=("poster_path",))
        # Include dynamically discovered titles when changing embedding models too.
        pending = conn.execute("""
            SELECT * FROM titles
            WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?
        """, (embedding_service.model_key,)).fetchall()
        if pending:
            conn.execute("DELETE FROM for_you_cache")

    for start in range(0, len(pending), 64):
        batch = pending[start:start + 64]
        texts = [embedding_service.build_title_embedding_text(dict(row)) for row in batch]
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
