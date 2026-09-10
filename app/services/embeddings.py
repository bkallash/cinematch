import logging
import hashlib
import re
import numpy as np
from typing import List, Optional
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    @property
    def is_local(self) -> bool:
        return settings.EMBEDDING_PROVIDER == "local" or not settings.OPENROUTER_API_KEY

    @property
    def model_key(self) -> str:
        model = "local:token-hash-v1" if self.is_local else f"openrouter:{settings.EMBEDDING_MODEL}"
        return f"{model}:{settings.EMBEDDING_DIM}"

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            # OpenRouter acts as an OpenAI-compatible endpoint
            api_key = settings.OPENROUTER_API_KEY or "dummy_key_for_init"
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.OPENROUTER_BASE_URL,
                timeout=30.0,
                max_retries=1,
            )
        return self._client

    async def get_embedding(self, text: str) -> np.ndarray:
        """Generate normalized float32 vector embedding for text."""
        return (await self.get_embeddings([text]))[0]

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch. Cloud failures never become stored local vectors."""
        if not texts:
            return []
        clean = [text.strip() or "cinema film" for text in texts]
        if self.is_local:
            return [self._local_embedding(text) for text in clean]
        response = await self._get_client().embeddings.create(
            model=settings.EMBEDDING_MODEL, input=clean, dimensions=settings.EMBEDDING_DIM,
        )
        items = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(clean))):
            raise ValueError("Embedding response did not match the requested batch")
        vectors = []
        for item in items:
            vec = np.asarray(item.embedding, dtype=np.float32)
            if vec.shape != (settings.EMBEDDING_DIM,) or not np.isfinite(vec).all() or np.linalg.norm(vec) == 0:
                raise ValueError("Invalid embedding dimensions or values")
            vectors.append(vec / np.linalg.norm(vec))
        return vectors

    @staticmethod
    def _local_embedding(text: str) -> np.ndarray:
        # Stable token overlap gives basic offline matching without a model download.
        stop_words = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "with", "for",
                      "title", "format", "genres", "director", "creator", "leading", "cast",
                      "plot", "atmospheric", "themes", "unknown"}
        vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
        for token in set(re.findall(r"\w+", text.casefold())) - stop_words:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % len(vec)] += 1.0 if digest[4] & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

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
