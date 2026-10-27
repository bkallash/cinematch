import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import get_db, init_db
from app.main import app
from app.services import llm
from app.services.embeddings import embedding_service
from app.services.tmdb import tmdb_service


@pytest.fixture(autouse=True)
def offline_db(tmp_path, monkeypatch):
    """Fresh database per test; no TMDB, no LLM, local embeddings."""
    monkeypatch.setattr(settings, "DATABASE_PATH", tmp_path / "test.db")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "TMDB_API_KEY", "")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(tmdb_service, "api_key", "")
    init_db()


@pytest.fixture
def client():
    # No `with`: skips the lifespan's starter-catalog seeding so tests control the data.
    return TestClient(app)


@pytest.fixture
def fake_llm(monkeypatch):
    """Enable the LLM paths and answer each call with `fake_llm.replies[<system prompt keyword>]`."""
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-key")
    calls = []

    async def complete_json(system, prompt, temperature):
        calls.append(prompt)
        key = next(k for k in complete_json.replies if k in system)
        return json.dumps(complete_json.replies[key])

    complete_json.replies, complete_json.calls = {}, calls
    monkeypatch.setattr(llm, "complete_json", complete_json)
    return complete_json


def add_title(title_id, name, axis=0, media="movie", genres=("Drama",), votes=5000, rating=7.5):
    """Insert a Title whose embedding is the unit vector on `axis`."""
    vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
    vec[axis] = 1
    with get_db() as conn:
        conn.execute("""
            INSERT INTO titles (id, tmdb_id, media_type, title, overview, genres, director_or_creator,
                                vote_average, vote_count, popularity, embedding, embedding_dim, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, 'Some Director', ?, ?, 50, ?, ?, ?)
        """, (title_id, title_id, media, name, f"Synopsis of {name}", json.dumps(list(genres)),
              rating, votes, vec.tobytes(), len(vec), embedding_service.model_key))


def rate(title_id, score):
    with get_db() as conn:
        conn.execute("INSERT INTO ratings (title_id, score) VALUES (?, ?)", (title_id, score))
