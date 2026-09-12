"""Rating-informed behavior through the approved Orchestrator and HTTP seams."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.config import settings
from app.database import get_db, init_db
from app.services.embeddings import embedding_service
from app.services.orchestrator import OrchestratorService


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_PATH", tmp_path / "ratings.db")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "TMDB_API_KEY", "")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    init_db()


def title(tid, axis=0, media="movie", genres=None, score=None, at="2026-09-01", model=None):
    vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
    vec[axis] = 1
    with get_db() as conn:
        conn.execute("""INSERT INTO titles
            (id, tmdb_id, title, media_type, overview, genres, embedding, embedding_dim,
             embedding_model, vote_average, vote_count, popularity, release_year)
            VALUES (?, ?, ?, ?, 'Canonical synopsis', ?, ?, ?, ?, 8, 1000, 50, 2020)""",
            (tid, tid, f"Title {tid}", media, json.dumps(genres or []), vec.tobytes(), len(vec),
             model or embedding_service.model_key))
        if score:
            conn.execute("INSERT INTO ratings (title_id, score, created_at, updated_at) VALUES (?, ?, ?, ?)",
                         (tid, score, at, at))


async def test_movie_favorites_outweigh_larger_conflicting_series_history():
    for tid in range(1, 4):
        title(tid, axis=0, score=6)
    for tid in range(4, 20):
        title(tid, axis=1, media="tv", score=6)
    title(30, axis=0)
    title(31, axis=1)
    result = await OrchestratorService().get_personalized_picks(media_type_preference="movie")
    assert result["picks"][0]["title_id"] == 30


async def test_distinct_interest_reaches_reranker_amid_redundant_dominant_interest(monkeypatch):
    for tid in range(1, 8):
        title(tid, axis=0, score=6)
    title(8, axis=1, score=6)
    for tid in range(20, 40):
        title(tid, axis=0)
    title(40, axis=1)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "provider-double")
    captured = []

    async def provider(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        tail = prompt.split("### Candidate", 1)[1]
        candidates, _ = json.JSONDecoder().raw_decode(tail[tail.index("["):])
        captured.extend(c["title_id"] for c in candidates)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            {"picks": [], "rejected_title_ids": captured})))])

    service = OrchestratorService()
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=provider))))
    from app.services.taste_dossier import taste_dossier_service
    # The same provider boundary also handles Dossier generation.
    monkeypatch.setattr(taste_dossier_service, "_client", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"full_summary": "Training evidence"}'))]))))))
    result = await service.get_personalized_picks()
    assert 40 in captured and any(tid in captured for tid in range(20, 40))
    assert len(captured) <= 15
    assert result["picks"] == []


async def test_genres_are_independent_positive_evidence_without_a_dislike_genre_ban():
    title(1, axis=0, genres=["Romance"], score=6)
    title(2, axis=1, genres=["Romance"], score=1)
    title(10, axis=2, genres=["Horror"])
    title(11, axis=2, genres=["Romance"])
    title(12, axis=1, genres=["Romance"])
    result = await OrchestratorService().get_personalized_picks()
    ids = [p["title_id"] for p in result["picks"]]
    assert ids.index(11) < ids.index(10)
    assert ids.index(11) < ids.index(12)
    assert 12 in ids


async def test_missing_embeddings_use_genres_and_cache_an_honest_shelf(monkeypatch):
    title(1, genres=["Romance"], score=6, model="old-schema")
    title(2, genres=["Horror"], model="old-schema")
    title(3, genres=["Romance"], model="old-schema")
    service = OrchestratorService()
    first = await service.get_personalized_picks()
    assert first["picks"][0]["title_id"] == 3
    assert "rated everything" not in first["message"].lower()


async def test_rebuilt_synopsis_evidence_outweighs_shared_people_and_year():
    from app.services.seeder import seed_starter_catalog
    for tid in (90001, 90002, 90003):
        title(tid, score=6 if tid == 90001 else None, model="old-schema")
    people = " ".join(f"Person{n}" for n in range(60))
    with get_db() as conn:
        conn.execute("UPDATE titles SET overview='quasar xenolinguist exoplanet' WHERE id IN (90001,90003)")
        conn.execute("UPDATE titles SET overview='couple courtship marriage' WHERE id=90002")
        conn.execute("UPDATE titles SET director_or_creator=?, cast_top=? WHERE id IN (90001,90002)",
                     (people, json.dumps([people])))
        conn.execute("UPDATE titles SET release_year=1960 WHERE id=90003")
        conn.execute("INSERT INTO watchlist (title_id) VALUES (90003)")
    await seed_starter_catalog()
    result = await OrchestratorService().get_personalized_picks(media_type_preference="movie")
    assert result["picks"][0]["title_id"] == 90003
    assert result["picks"][0]["is_on_watchlist"]
    assert result["ratings_count"] == 1
    assert 90001 not in [p["title_id"] for p in result["picks"]]


async def test_shortlist_prefers_a_related_variation_over_an_identical_second_title():
    title(1, axis=0, score=6)
    title(10, axis=0)
    title(11, axis=0)
    title(12, axis=1)
    vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
    vec[:2] = [0.95, (1 - 0.95 ** 2) ** 0.5]
    with get_db() as conn:
        conn.execute("UPDATE titles SET embedding=? WHERE id=12", (vec.tobytes(),))
    result = await OrchestratorService().get_personalized_picks(limit=3)
    assert [p["title_id"] for p in result["picks"]][:2] == [10, 12]


async def test_weak_large_pool_discovers_once_and_caches_the_result(monkeypatch):
    from app.services.tmdb import tmdb_service
    title(1, axis=0, genres=["Romance"], score=6)
    for tid in range(10, 40):
        title(tid, axis=1)
    item = {"tmdb_id": 50, "media_type": "movie", "title": "Discovered romance",
            "overview": "Canonical synopsis", "genres": ["Romance"], "release_year": 2020}
    discover = AsyncMock(return_value=[item])
    monkeypatch.setattr(settings, "TMDB_API_KEY", "provider-double")
    monkeypatch.setattr(tmdb_service, "discover_titles", discover)
    monkeypatch.setattr(tmdb_service, "get_title_details", AsyncMock(return_value=item))
    vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
    vec[0] = 1
    monkeypatch.setattr(embedding_service, "get_embedding", AsyncMock(return_value=vec))
    service = OrchestratorService()
    first = await service.get_personalized_picks(media_type_preference="movie")
    second = await service.get_personalized_picks(media_type_preference="movie")
    assert first["picks"][0]["tmdb_id"] == 50
    assert [p["title_id"] for p in first["picks"]] == [p["title_id"] for p in second["picks"]]
    assert discover.await_count == 1
    assert discover.call_args.kwargs["with_genres"] == "10749"


@pytest.mark.parametrize("history", ["none", "negative", "missing"])
async def test_sparse_shelves_survive_discovery_outage_and_are_cached(monkeypatch, history):
    from app.services.tmdb import tmdb_service
    if history != "none":
        title(1, score=1 if history == "negative" else 6,
              model="old-schema" if history == "missing" else None)
    title(2, axis=1)
    discover = AsyncMock(side_effect=RuntimeError("Provider unavailable"))
    monkeypatch.setattr(settings, "TMDB_API_KEY", "provider-double")
    monkeypatch.setattr(tmdb_service, "discover_titles", discover)
    service = OrchestratorService()
    first = await service.get_personalized_picks(media_type_preference="movie")
    second = await service.get_personalized_picks(media_type_preference="movie")
    assert [p["title_id"] for p in first["picks"]] == [2]
    assert first["picks"] == [{k: v for k, v in p.items() if k != "user_rating"} for p in second["picks"]]
    assert discover.await_count == 1
    if history == "negative":
        assert "dislikes" in first["message"] and "love" not in first["message"]
    if history == "none":
        assert not first["personalized"] and "popular" in first["message"]


async def test_empty_shelf_is_cached_and_embedding_identity_invalidates_it(monkeypatch):
    from app.services.tmdb import tmdb_service
    discover = AsyncMock(return_value=[])
    monkeypatch.setattr(settings, "TMDB_API_KEY", "provider-double")
    monkeypatch.setattr(tmdb_service, "discover_titles", discover)
    service = OrchestratorService()
    for _ in range(2):
        assert (await service.get_personalized_picks(media_type_preference="movie"))["picks"] == []
    assert discover.await_count == 1
    monkeypatch.setattr(settings, "EMBEDDING_DIM", settings.EMBEDDING_DIM + 1)
    assert (await service.get_personalized_picks(media_type_preference="movie"))["picks"] == []
    assert discover.await_count == 2


@pytest.mark.parametrize("media", ["movie", "tv"])
def test_for_you_skip_refills_and_persists_without_rating(media):
    from fastapi.testclient import TestClient
    from app.main import app
    for tid in range(1, 8):
        title(tid, media=media)
    client = TestClient(app)
    url = f"/api/for-you?media_type={media}&format=json"
    initial = client.get(url).json()["picks"]
    skipped = initial[0]["title_id"]
    with get_db() as conn:
        conn.execute("INSERT INTO watchlist (title_id) VALUES (?)", (skipped,))
    response = client.post(f"/api/for-you/skip/{skipped}?media_type={media}")
    assert response.status_code == 200
    assert 'aria-label="Skip ' in response.text
    assert f'aria-label="Skip Title {skipped}"' not in response.text
    for suffix in ("", "&refresh=true"):
        picks = client.get(url + suffix).json()["picks"]
        assert len(picks) == 5
        assert skipped not in {p["title_id"] for p in picks}
        assert all(p["media_type"] == media for p in picks)
    assert client.post(f"/api/for-you/skip/{skipped}").status_code == 200
    assert client.post("/api/for-you/skip/9999").status_code == 404
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM skipped_titles").fetchone()[0] == 0
        assert conn.execute("SELECT title_id FROM watchlist").fetchone()[0] == skipped


def test_http_rating_edit_refreshes_next_shelf_without_manual_refresh():
    import asyncio
    from fastapi.testclient import TestClient
    from app.main import app
    title(1, axis=0)
    title(2, axis=0)
    title(3, axis=1)
    with get_db() as conn:
        for tid, synopsis in [(1, "astronaut quasar voyager"), (2, "astronaut quasar voyager"),
                              (3, "romantic couple courtship")]:
            vec = asyncio.run(embedding_service.get_embedding(synopsis))
            conn.execute("UPDATE titles SET overview=?, embedding=? WHERE id=?", (synopsis, vec.tobytes(), tid))
    client = TestClient(app)
    assert client.post("/api/ratings", json={"title_id": 1, "score": 6}).status_code == 200
    liked = client.get("/api/for-you?format=json").json()
    assert liked["picks"][0]["title_id"] == 2
    assert client.post("/api/ratings", json={"title_id": 1, "score": 1, "notes": "The ending disappointed me"}).status_code == 200
    disliked = client.get("/api/for-you?format=json").json()
    assert disliked["picks"][0]["title_id"] == 3
    assert all(p["title_id"] != 1 for p in disliked["picks"])


async def test_stronger_rating_and_recent_evidence_win_comparable_candidates():
    title(1, axis=0, score=4)
    title(2, axis=1, score=6)
    title(10, axis=0)
    title(11, axis=1)
    service = OrchestratorService()
    assert (await service.get_personalized_picks())["picks"][0]["title_id"] == 11
    with get_db() as conn:
        conn.execute("UPDATE ratings SET score=6, created_at='2020-01-01', updated_at='2020-01-01' WHERE title_id=2")
        conn.execute("UPDATE ratings SET score=6 WHERE title_id=1")
    assert (await service.get_personalized_picks())["picks"][0]["title_id"] == 10


async def test_sparse_media_uses_cross_media_history_until_its_own_evidence_exists():
    title(1, axis=0, media="tv", score=6)
    title(10, axis=1)
    title(11, axis=0)
    service = OrchestratorService()
    assert (await service.get_personalized_picks(media_type_preference="movie"))["picks"][0]["title_id"] == 11
    title(2, axis=1, score=6)
    assert (await service.get_personalized_picks(media_type_preference="movie"))["picks"][0]["title_id"] == 10


async def test_vibe_overrides_taste_and_constraints_exclude_watchlist_and_session_repeats(monkeypatch):
    from app.services.taste_dossier import taste_dossier_service
    title(1, axis=0, score=6)
    for tid in range(2, 7):
        title(tid, axis=0 if tid == 3 else 1, genres=["Drama"])
    with get_db() as conn:
        conn.execute("UPDATE titles SET release_year=1995, director_or_creator='Ada Director'")
        conn.execute("UPDATE titles SET release_year=2020 WHERE id=4")
        conn.execute("UPDATE titles SET director_or_creator='Different Person' WHERE id=5")
        conn.execute("UPDATE titles SET genres='[\"Drama\",\"Horror\"]' WHERE id=6")
        conn.execute("INSERT INTO watchlist (title_id) VALUES (1),(4)")

    async def provider(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "### Candidate" in prompt:
            raise RuntimeError("Reranker unavailable")
        data = {"semantic_vibe": "a thoughtful drama", "media_type": "movie", "genres": ["Drama"],
                "excluded_genres": ["Horror"], "person": "Ada Director", "year_min": 1990, "year_max": 1999}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

    service = OrchestratorService()
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=provider))))
    monkeypatch.setattr(taste_dossier_service, "_client", service._client)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "provider-double")
    query = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
    query[1] = 1
    monkeypatch.setattr(embedding_service, "get_embedding", AsyncMock(return_value=query))
    first = await service.handle_vibe_query("A 1990s drama by Ada Director, no horror", "constraints", "movie")
    assert [p["title_id"] for p in first["recommendations"]] == [2, 3]
    second = await service.handle_vibe_query("More with the same constraints", "constraints", "movie")
    assert second["recommendations"] == []


@pytest.mark.parametrize("similarity", [0.01, 0.95])
async def test_uncalibrated_cloud_evidence_allows_only_one_discovery_round(monkeypatch, similarity):
    from app.services.tmdb import tmdb_service
    from app.services.taste_dossier import taste_dossier_service
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "provider-double")
    monkeypatch.setattr(settings, "TMDB_API_KEY", "provider-double")
    title(1, axis=0, score=6)
    for tid in range(10, 30):
        title(tid, axis=1)
        vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
        vec[:2] = [similarity, (1 - similarity ** 2) ** 0.5]
        with get_db() as conn:
            conn.execute("UPDATE titles SET embedding=? WHERE id=?", (vec.tobytes(), tid))
    discover = AsyncMock(return_value=[])
    monkeypatch.setattr(tmdb_service, "discover_titles", discover)
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=RuntimeError("Reranker unavailable")))))
    service = OrchestratorService()
    service._client = provider
    monkeypatch.setattr(taste_dossier_service, "_client", provider)
    first = await service.get_personalized_picks(media_type_preference="movie")
    second = await service.get_personalized_picks(media_type_preference="movie")
    assert len(first["picks"]) == 5
    assert [p["title_id"] for p in first["picks"]] == [p["title_id"] for p in second["picks"]]
    assert discover.await_count == 1
