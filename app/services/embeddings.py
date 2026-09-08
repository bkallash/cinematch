import logging
import numpy as np
from typing import List, Optional, Union
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.model = settings.EMBEDDING_MODEL
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            # OpenRouter acts as an OpenAI-compatible endpoint
            api_key = settings.OPENROUTER_API_KEY or "dummy_key_for_init"
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.OPENROUTER_BASE_URL
            )
        return self._client

    async def get_embedding(self, text: str) -> np.ndarray:
        """Generate normalized float32 vector embedding for text."""
        clean_text = text.strip()
        if not clean_text:
            clean_text = "cinema film"

        if not settings.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set. Generating deterministic pseudo-embedding.")
            # Deterministic hash-based pseudo vector for testing without active API key
            rng = np.random.default_rng(abs(hash(clean_text)) % (2**32))
            vec = rng.standard_normal(settings.EMBEDDING_DIM).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm if norm > 0 else 1.0)

        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=clean_text
            )
            embedding_data = response.data[0].embedding
            vec = np.array(embedding_data, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            logger.error(f"Error calling embedding API: {e}. Falling back to hash vector.")
            rng = np.random.default_rng(abs(hash(clean_text)) % (2**32))
            vec = rng.standard_normal(settings.EMBEDDING_DIM).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm if norm > 0 else 1.0)

    @staticmethod
    def build_title_embedding_text(title_data: dict) -> str:
        """Compose a high-signal semantic string for a Title."""
        title = title_data.get("title", "")
        year = title_data.get("release_year") or "Unknown"
        media_type = "Movie" if title_data.get("media_type") == "movie" else "TV Series"
        genres = ", ".join(title_data.get("genres") or [])
        director = title_data.get("director_or_creator") or "Unknown"
        cast = ", ".join((title_data.get("cast_top") or [])[:4])
        overview = title_data.get("overview") or ""

        return (
            f"Title: {title} ({year})\n"
            f"Format: {media_type}\n"
            f"Genres: {genres}\n"
            f"Director/Creator: {director}\n"
            f"Leading Cast: {cast}\n"
            f"Plot & Atmospheric Themes: {overview}"
        )

    @staticmethod
    def vec_to_bytes(vec: np.ndarray) -> bytes:
        """Convert numpy array to SQLite BLOB bytes."""
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_vec(blob: bytes) -> np.ndarray:
        """Convert SQLite BLOB bytes back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between a normalized query vector (D,)
        and a normalized document matrix (N, D).
        Returns a 1D array of scores in [-1, 1].
        """
        if doc_matrix.ndim == 1:
            return np.array([float(np.dot(query_vec, doc_matrix))])
        return np.dot(doc_matrix, query_vec)

embedding_service = EmbeddingService()
