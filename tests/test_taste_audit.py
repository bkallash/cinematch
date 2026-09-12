"""Behavioral checks for taste evidence, conversation continuity and abstention."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.database import get_db, init_db
from app.services.embeddings import embedding_service
from app.services.orchestrator import OrchestratorService
from app.services.taste_dossier import TasteDossierService
from app.services.tmdb import tmdb_service


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_PATH", tmp_path / "audit.db")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "TMDB_API_KEY", "")
    init_db()


def title(title_id, name, axis=0, genres=None):
    vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
    vec[axis] = 1
    with get_db() as conn:
        conn.execute("""INSERT INTO titles
            (id, tmdb_id, media_type, title, overview, genres, embedding, embedding_dim, embedding_model)
            VALUES (?, ?, 'movie', ?, ?, ?, ?, ?, ?)""",
            (title_id, title_id, name, f"Synopsis for {name}", json.dumps(genres or ["Drama"]),
             vec.tobytes(), len(vec), embedding_service.model_key))


def chat_mock(service, data):
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(data)))]))
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return create


async def test_follow_up_contains_ordered_recommendation_titles(monkeypatch):
    title(1, "First Pick Sentinel")
    title(2, "Second Pick Sentinel")
    with get_db() as conn:
        conn.execute("""INSERT INTO chat_messages (session_id, role, content, recommended_title_ids)
            VALUES ('follow-up', 'assistant', 'Here are your picks', '[2, 1]')""")
    svc = OrchestratorService()
    parse = AsyncMock(return_value={"semantic_vibe": "drama"})
    monkeypatch.setattr(svc, "_parse_query_intent", parse)
    await svc.handle_vibe_query("More like the first pick", "follow-up")
    history = json.dumps(parse.call_args.args[2])
    assert "Second Pick Sentinel" in history
    assert history.index("Second Pick Sentinel") < history.index("First Pick Sentinel")


async def test_chat_retrieval_uses_ratings_for_equally_matching_vibes(monkeypatch):
    title(1, "Disliked", axis=0)
    title(2, "Loved", axis=1)
    title(3, "Bad Fit", axis=0)
    title(4, "Good Fit", axis=1)
    with get_db() as conn:
        conn.execute("INSERT INTO ratings (title_id, score) VALUES (1, 1), (2, 6)")
    query = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
    query[:2] = 2 ** -0.5
    monkeypatch.setattr(embedding_service, "get_embedding", AsyncMock(return_value=query))
    result = await OrchestratorService().handle_vibe_query("something engaging")
    assert result["recommendations"][0]["title_id"] == 4


@pytest.mark.parametrize("surface", ["chat", "shelf"])
async def test_rerank_receives_negative_notes_even_before_dossier_threshold(monkeypatch, surface):
    title(1, "Disliked Sentinel")
    title(2, "Candidate")
    with get_db() as conn:
        conn.execute("""INSERT INTO ratings (title_id, score, aspect_tags, notes)
            VALUES (1, 2, '["Ending"]', 'Unresolved endings spoil the story for me')""")
    svc = OrchestratorService()
    rows = await svc._search_local_candidates({"semantic_vibe": "drama"})
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    create = chat_mock(svc, {"picks": [], "recommendations": []})
    if surface == "chat":
        await svc._rerank_and_justify("drama", [], rows, {})
    else:
        await svc._rerank_personalized_picks(rows, {}, 1, 1)
    prompt = json.dumps(create.call_args.kwargs["messages"])
    assert "Unresolved endings spoil the story for me" in prompt
    assert "Disliked Sentinel" in prompt


@pytest.mark.parametrize("surface", ["chat", "shelf"])
async def test_explicitly_rejected_candidates_are_never_backfilled(monkeypatch, surface):
    for i in range(1, 5):
        title(i, f"Candidate {i}")
    svc = OrchestratorService()
    rows = await svc._search_local_candidates({"semantic_vibe": "drama"})
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    chat_mock(svc, {"picks": [], "recommendations": [], "rejected_title_ids": [1, 2, 3, 4]})
    if surface == "chat":
        result = (await svc._rerank_and_justify("no gore", [], rows, {}))["recommendations"]
    else:
        result = await svc._rerank_personalized_picks(rows, {}, 3, 4)
    assert result == []


def test_fallback_reason_does_not_invent_dossier_alignment():
    reason = OrchestratorService()._generate_personalized_reason(
        {"title": "Slasher", "genres": ["Horror"], "overview": "A masked killer attacks."},
        dossier={"atmospheric_preferences": ["gentle, gore-free comfort"]})
    assert "gentle, gore-free comfort" not in reason
    assert "acclaimed" not in reason


async def test_dossier_preserves_old_extremes_and_supplies_plot_evidence(monkeypatch):
    for i in range(1, 207):
        title(i, "Old Favorite Sentinel" if i == 1 else "Old Dislike Sentinel" if i == 2 else f"Recent {i}")
    with get_db() as conn:
        conn.execute("""INSERT INTO ratings (title_id, score, created_at, updated_at)
            SELECT id, CASE WHEN id=1 THEN 6 WHEN id=2 THEN 1 ELSE 4 END,
                CASE WHEN id<3 THEN '2020-01-01' ELSE '2026-09-01' END,
                CASE WHEN id<3 THEN '2020-01-01' ELSE '2026-09-01' END FROM titles""")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = TasteDossierService()
    create = chat_mock(svc, {"full_summary": "Evidence-based summary"})
    await svc.get_or_update_dossier()
    prompt = json.dumps(create.call_args.kwargs["messages"])
    assert "Old Favorite Sentinel" in prompt and "Old Dislike Sentinel" in prompt
    assert "Synopsis for Old Favorite Sentinel" in prompt


def test_mixed_media_genre_filter_accepts_equivalent_tv_taxonomy():
    assert OrchestratorService._matches_intent(
        {"media_type": "tv", "genres": ["Sci-Fi & Fantasy"]}, {"genres": ["Science Fiction"]})
    assert not OrchestratorService._matches_intent(
        {"media_type": "movie", "genres": ["Horror"]}, {"excluded_genres": ["Horror"]})


async def test_embedding_outage_still_allows_metadata_reranking(monkeypatch):
    title(1, "Available Metadata")
    with get_db() as conn:
        conn.execute("UPDATE titles SET embedding=NULL, embedding_model=NULL")
    monkeypatch.setattr(embedding_service, "get_embedding", AsyncMock(side_effect=RuntimeError("offline")))
    result = await OrchestratorService().handle_vibe_query("drama")
    assert [r["title_id"] for r in result["recommendations"]] == [1]


def test_rating_http_round_trip_preserves_and_replaces_qualitative_evidence(monkeypatch):
    title(1, "Rated")
    monkeypatch.setattr(main, "_ensure_title_embedding", AsyncMock())
    client = TestClient(main.app)
    for score, note in [(6, "Loved the deliberate pacing"), (2, "Changed my mind: too slow")]:
        assert client.post("/api/ratings", json={"title_id": 1, "score": score,
            "aspect_tags": ["Pacing"], "notes": note}).status_code == 200
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM ratings").fetchall()
            assert len(rows) == 1
            assert (rows[0]["score"], rows[0]["notes"]) == (score, note)
            assert json.loads(rows[0]["aspect_tags"]) == ["Pacing"]


async def test_mixed_media_discovery_searches_movies_and_series(monkeypatch):
    discover = AsyncMock(return_value=[])
    monkeypatch.setattr(tmdb_service, "discover_titles", discover)
    await OrchestratorService()._discover_and_cache_tmdb({"genres": ["Science Fiction"]})
    calls = {c.kwargs["media_type"]: c.kwargs["with_genres"] for c in discover.call_args_list}
    assert calls == {"movie": "878", "tv": "10765"}


async def test_short_suitable_shelf_is_cached_without_repeated_model_calls(monkeypatch):
    for i in range(1, 5):
        title(i, f"Title {i}")
    with get_db() as conn:
        conn.execute("INSERT INTO ratings (title_id,score) VALUES (1,6)")
    # Keep stored local embeddings compatible when enabling only the reranker.
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = OrchestratorService()
    create = chat_mock(svc, {"picks": [{"title_id": 2, "title": "Title 2", "reason": "A grounded candidate worth exploring."}],
                             "rejected_title_ids": [3, 4]})
    first = await svc.get_personalized_picks(limit=3)
    second = await svc.get_personalized_picks(limit=3)
    assert [p["title_id"] for p in first["picks"]] == [2]
    assert [(p["title_id"], p["reason"]) for p in second["picks"]] == [(p["title_id"], p["reason"]) for p in first["picks"]]
    assert create.await_count == 1


async def test_enrichment_persists_the_same_metadata_used_for_embedding(monkeypatch):
    title(1, "Enriched")
    monkeypatch.setattr(settings, "TMDB_API_KEY", "test-only")
    monkeypatch.setattr(tmdb_service, "get_title_details", AsyncMock(return_value={
        "overview": "Updated synopsis with a warm ensemble", "genres": ["Comedy"],
        "director_or_creator": "A Creator", "cast_top": ["An Actor"],
    }))
    await main._ensure_title_embedding(1)
    with get_db() as conn:
        row = conn.execute("SELECT overview,genres FROM titles WHERE id=1").fetchone()
    assert row["overview"] == "Updated synopsis with a warm ensemble"
    assert json.loads(row["genres"]) == ["Comedy"]


@pytest.mark.parametrize("surface", ["chat", "shelf"])
async def test_model_cannot_attach_another_titles_reason_to_a_card(monkeypatch, surface):
    title(301, "Her")
    title(902, "Lost in Translation")
    svc = OrchestratorService()
    rows = await svc._search_local_candidates({"semantic_vibe": "romance"})
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    wrong = {"title_id": 301, "title": "Lost in Translation", "reason": "WRONG_CARD_SENTINEL: Two lonely travelers meet in Tokyo."}
    chat_mock(svc, {"picks": [wrong], "recommendations": [wrong]})
    if surface == "chat":
        picks = (await svc._rerank_and_justify("romance", [], rows, {}))["recommendations"]
    else:
        picks = await svc._rerank_personalized_picks(rows, {}, 2, 2)
    assert picks
    assert all("WRONG_CARD_SENTINEL" not in p["reason"] for p in picks)


def test_regenerated_dossier_renders_model_text_as_text(monkeypatch):
    monkeypatch.setattr(main.taste_dossier_service, "get_or_update_dossier", AsyncMock(return_value={
        "full_summary": "<script>untrusted()</script>", "core_loves": ["<img src=x onerror=untrusted()>"],
        "deal_breakers": [], "atmospheric_preferences": [], "creator_affinities": [],
    }))
    response = TestClient(main.app).post("/api/dossier/regenerate")
    assert response.status_code == 200
    assert "<script>untrusted()" not in response.text
    assert "&lt;script&gt;" in response.text


@pytest.mark.parametrize("surface", ["chat", "shelf"])
async def test_valid_model_shortlist_is_not_padded_with_unselected_titles(monkeypatch, surface):
    for i in range(1, 5):
        title(i, f"Candidate {i}")
    svc = OrchestratorService()
    rows = await svc._search_local_candidates({"semantic_vibe": "drama"})
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    selected = {"title_id": 1, "title": "Candidate 1", "reason": "The only candidate supported by the available evidence."}
    chat_mock(svc, {"picks": [selected], "recommendations": [selected], "rejected_title_ids": []})
    if surface == "chat":
        picks = (await svc._rerank_and_justify("specific mood", [], rows, {}))["recommendations"]
    else:
        picks = await svc._rerank_personalized_picks(rows, {}, 3, 4)
    assert [p["title_id"] for p in picks] == [1]
