import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import httpx
import pytest

from app.config import settings
from app.database import get_db, init_db
from app.services.embeddings import embedding_service
from app.services.orchestrator import OrchestratorService
from app.services.taste_dossier import TasteDossierService
from app.services.seeder import seed_starter_catalog
from app.services.tmdb import tmdb_service


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_PATH", tmp_path / "test.db")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "TMDB_API_KEY", "")
    init_db()


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


def mock_chat(service, data):
    create = AsyncMock(return_value=response(data))
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return create


async def candidates():
    vec = await embedding_service.get_embedding("romance")
    with get_db() as conn:
        for i in range(1, 10):
            conn.execute(
                "INSERT INTO titles (tmdb_id, media_type, title, genres, director_or_creator, embedding, embedding_dim) VALUES (?, 'movie', ?, ?, ?, ?, ?)",
                (i, f"Title {i}", '["Romance"]' if i == 1 else '["Horror"]',
                 "Requested Director" if i == 1 else "Wrong Director", vec.tobytes(), len(vec)),
            )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(titles)")}
        if "embedding_model" in cols:
            conn.execute("UPDATE titles SET embedding_model = ?", (embedding_service.model_key,))
    return await OrchestratorService()._search_local_candidates({"semantic_vibe": "romance"})


def test_local_embedding_stable_across_processes():
    code = "import asyncio,hashlib; from app.config import settings; settings.OPENROUTER_API_KEY=''; from app.services.embeddings import embedding_service; print(hashlib.sha256(asyncio.run(embedding_service.get_embedding('romance cinema')).tobytes()).hexdigest())"
    values = [subprocess.check_output([sys.executable, "-B", "-c", code],
              env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True).strip() for seed in (1, 2)]
    assert values[0] == values[1]


@pytest.mark.asyncio
async def test_explicit_filters():
    await candidates()
    result = await OrchestratorService()._search_local_candidates({
        "semantic_vibe": "romance", "genres": ["Romance"], "person": "Requested Director",
    })
    assert [r["title_id"] for r in result] == [1]


@pytest.mark.asyncio
async def test_chat_history_and_unique_bounded_picks(monkeypatch):
    rows = await candidates()
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = OrchestratorService()
    create = mock_chat(svc, {"recommendations": [{"title_id": 1, "reason": "A warm romantic story."}] * 7})
    result = await svc._rerank_and_justify("More like that", [{"role": "user", "content": "HISTORY_SENTINEL"}], rows, {})
    assert "HISTORY_SENTINEL" in json.dumps(create.call_args.kwargs["messages"])
    ids = [r["title_id"] for r in result["recommendations"]]
    assert 3 <= len(ids) <= 5
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_personalized_duplicates_are_backfilled(monkeypatch):
    rows = await candidates()
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = OrchestratorService()
    mock_chat(svc, {"picks": [{"title_id": 1, "reason": "A warm romantic story."}] * 5})
    picks = await svc._rerank_personalized_picks(rows, {}, 3, 5)
    assert len({p["title_id"] for p in picks}) == 5


@pytest.mark.asyncio
async def test_bad_intent_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = OrchestratorService()
    mock_chat(svc, {"semantic_vibe": 42, "genres": "Romance"})
    intent = await svc._parse_query_intent("romance", "movie")
    assert intent["semantic_vibe"] == "romance"
    assert intent["media_type"] == "movie"


@pytest.mark.asyncio
async def test_rating_edit_during_synthesis_stays_dirty(monkeypatch):
    await candidates()
    with get_db() as conn:
        for tid in (1, 2, 3):
            conn.execute("INSERT INTO ratings (title_id, score) VALUES (?, 6)", (tid,))
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = TasteDossierService()
    create = mock_chat(svc, {"full_summary": "Updated taste"})

    async def edit_during_request(**kwargs):
        with get_db() as conn:
            conn.execute("UPDATE ratings SET score=1 WHERE title_id=1")
        svc.mark_dirty()
        return response({"full_summary": "Old taste"})

    create.side_effect = edit_during_request
    await svc.get_or_update_dossier()
    with get_db() as conn:
        assert conn.execute("SELECT is_dirty FROM taste_dossiers").fetchone()[0] == 1
    create.side_effect = None
    await svc.get_or_update_dossier()
    assert create.await_count == 2


async def test_local_provider_never_calls_cloud(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    create = AsyncMock(side_effect=AssertionError("Local mode contacted cloud"))
    monkeypatch.setattr(embedding_service, "_client", SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    vec = await embedding_service.get_embedding("romance comedy")
    assert np.isclose(np.linalg.norm(vec), 1)
    create.assert_not_called()


async def test_cloud_failure_does_not_persist_fallback(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    create = AsyncMock(side_effect=RuntimeError("Provider unavailable"))
    monkeypatch.setattr(embedding_service, "_client", SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError):
        await embedding_service.get_embedding("romance")
    await seed_starter_catalog()
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM titles WHERE embedding IS NOT NULL").fetchone()[0] == 0


async def test_embedding_batch_orders_and_validates_results(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr(settings, "EMBEDDING_DIM", 2)
    create = AsyncMock(return_value=SimpleNamespace(data=[
        SimpleNamespace(index=1, embedding=[0, 2]), SimpleNamespace(index=0, embedding=[2, 0]),
    ]))
    monkeypatch.setattr(embedding_service, "_client", SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    vectors = await embedding_service.get_embeddings(["first", "second"])
    assert np.array_equal(vectors, [[1, 0], [0, 1]])
    assert create.await_count == 1
    create.return_value.data[0].embedding = [0, 1, 2]
    with pytest.raises(ValueError):
        await embedding_service.get_embeddings(["first", "second"])


async def test_incompatible_vectors_excluded_and_rebuilt():
    await candidates()
    with get_db() as conn:
        conn.execute("INSERT INTO ratings (title_id, score) VALUES (1,6), (2,6)")
        conn.execute("UPDATE titles SET embedding = ?, embedding_model = 'old-model' WHERE id = 2", (np.ones(2, dtype=np.float32).tobytes(),))
        conn.execute("UPDATE titles SET embedding_model = 'old-model' WHERE id = 3")
    svc = OrchestratorService()
    assert svc._build_taste_profile()["taste_vector"].shape == (settings.EMBEDDING_DIM,)
    result = await svc._search_local_candidates({"semantic_vibe": "romance"})
    assert 3 not in [r["title_id"] for r in result]
    await seed_starter_catalog()
    with get_db() as conn:
        row = conn.execute("SELECT embedding, embedding_model FROM titles WHERE id=2").fetchone()
    assert row["embedding_model"] == embedding_service.model_key
    assert len(embedding_service.bytes_to_vec(row["embedding"])) == settings.EMBEDDING_DIM


async def test_concurrent_dossier_reads_share_synthesis(monkeypatch):
    await candidates()
    with get_db() as conn:
        conn.execute("INSERT INTO ratings (title_id, score) VALUES (1,6), (2,5), (3,4)")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = TasteDossierService()
    create = mock_chat(svc, {})
    entered, release = asyncio.Event(), asyncio.Event()

    async def complete(**kwargs):
        entered.set()
        await release.wait()
        return response({"full_summary": "Enjoys detailed storytelling."})

    create.side_effect = complete
    first = asyncio.create_task(svc.get_or_update_dossier())
    await entered.wait()
    second = asyncio.create_task(svc.get_or_update_dossier())
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, second)
    assert results[0]["full_summary"] == results[1]["full_summary"]
    assert create.await_count == 1


async def test_chat_reuses_embedding_after_discovery_and_supplies_intent_history(monkeypatch):
    svc = OrchestratorService()
    monkeypatch.setattr(settings, "TMDB_API_KEY", "test-only")
    parse = AsyncMock(return_value={"semantic_vibe": "romance"})
    embed = AsyncMock(return_value=np.zeros(settings.EMBEDDING_DIM, dtype=np.float32))
    search = AsyncMock(return_value=[])
    monkeypatch.setattr(svc, "_parse_query_intent", parse)
    monkeypatch.setattr(svc, "_search_local_candidates", search)
    monkeypatch.setattr(svc, "_discover_and_cache_tmdb", AsyncMock(return_value=[1]))
    monkeypatch.setattr(embedding_service, "get_embedding", embed)
    with get_db() as conn:
        conn.execute("INSERT INTO chat_messages (session_id,role,content) VALUES ('test','user','Earlier request')")
    await svc.handle_vibe_query("More like that", "test")
    assert parse.call_args.args[2] == [{"role": "user", "content": "Earlier request"}]
    assert embed.await_count == 1
    assert search.await_count == 2


async def test_stale_for_you_write_is_not_reused():
    await candidates()
    svc = OrchestratorService()
    await svc.get_personalized_picks(limit=3)
    with get_db() as conn:
        cached = dict(conn.execute("SELECT * FROM for_you_cache").fetchone())
        conn.execute("INSERT INTO ratings (title_id, score) VALUES (1, 6)")
    # Reproduce a generation that finishes after the rating's cache invalidation.
    svc._save_for_you_cache("default_user", "movie", json.loads(cached["picks_json"]), "STALE", True,
                            1, None, cached["ratings_revision"])
    result = await svc.get_personalized_picks(limit=3)
    assert result["message"] != "STALE"
    assert 1 not in [p["title_id"] for p in result["picks"]]


async def test_discovery_respects_genre_and_repairs_existing_title(monkeypatch):
    await candidates()
    with get_db() as conn:
        conn.execute("UPDATE titles SET embedding = NULL, embedding_model = NULL WHERE id = 1")
    item = {"tmdb_id": 1, "media_type": "movie", "title": "Title 1", "genres": ["Romance"],
            "release_year": 2000, "director_or_creator": "Requested Director", "cast_top": []}
    discover = AsyncMock(return_value=[item])
    monkeypatch.setattr(tmdb_service, "discover_titles", discover)
    monkeypatch.setattr(tmdb_service, "get_title_details", AsyncMock(return_value=item))
    svc = OrchestratorService()
    inserted = await svc._discover_and_cache_tmdb({"genres": ["Romance"], "media_type": "movie"})
    assert inserted == [1]
    assert discover.call_args.kwargs["with_genres"] == "10749"
    with get_db() as conn:
        assert conn.execute("SELECT embedding_model FROM titles WHERE id = 1").fetchone()[0] == embedding_service.model_key


async def test_person_discovery_uses_acting_and_directing_credits(monkeypatch):
    monkeypatch.setattr(tmdb_service, "api_key", "test-only")
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path.endswith("search/person"):
            return httpx.Response(200, json={"results": [{"id": 42, "name": "Requested Person"}]})
        common = {"media_type": "movie", "title": "Title", "genre_ids": [10749], "release_date": "2000-01-01"}
        return httpx.Response(200, json={
            "cast": [{**common, "id": 1}],
            "crew": [{**common, "id": 2, "job": "Director"}, {**common, "id": 3, "job": "Producer"}],
        })

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs))
    titles = await tmdb_service.find_person_titles("Requested Person", "movie")
    assert {t["tmdb_id"] for t in titles} == {1, 2}
    assert paths == ["/3/search/person", "/3/person/42/combined_credits"]


async def test_person_discovery_enriches_existing_cast(monkeypatch):
    await candidates()
    item = {"tmdb_id": 1, "media_type": "movie", "title": "Title 1", "genres": ["Romance"],
            "release_year": 2000, "director_or_creator": "Requested Director", "cast_top": ["Requested Actor"]}
    monkeypatch.setattr(tmdb_service, "find_person_titles", AsyncMock(return_value=[item]))
    monkeypatch.setattr(tmdb_service, "get_title_details", AsyncMock(return_value={**item, "cast_top": ["Lead Actor"]}))
    svc = OrchestratorService()
    intent = {"semantic_vibe": "romance", "genres": ["Romance"], "person": "Requested Actor", "media_type": "movie"}
    assert await svc._discover_and_cache_tmdb(intent) == [1]
    assert [t["title_id"] for t in await svc._search_local_candidates(intent)] == [1]


async def test_intent_prompt_receives_history(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = OrchestratorService()
    create = mock_chat(svc, {"semantic_vibe": "romance"})
    await svc._parse_query_intent("More like that", "movie", [{"role": "user", "content": "HISTORY_SENTINEL"}])
    assert "HISTORY_SENTINEL" in json.dumps(create.call_args.kwargs["messages"])


async def test_invalid_selection_entries_backfill_from_candidates(monkeypatch):
    rows = await candidates()
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-only")
    svc = OrchestratorService()
    mock_chat(svc, {"recommendations": [None, {"title_id": []}, {"title_id": 999}, {"title_id": 1, "reason": None}]})
    result = await svc._rerank_and_justify("romance", [], rows, {})
    ids = [r["title_id"] for r in result["recommendations"]]
    assert len(set(ids)) == 3
    assert set(ids).issubset({r["title_id"] for r in rows})
