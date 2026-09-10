import asyncio
import json
from starlette.testclient import TestClient
from app.database import init_db, get_db
from app.services.orchestrator import orchestrator_service
from app.main import app


async def test_for_you_cache():
    print("1. Initializing database and ensuring for_you_cache table exists...")
    init_db()

    with get_db() as conn:
        # Clear cache for clean test run
        conn.execute("DELETE FROM for_you_cache")
        # Ensure we have titles
        title_count = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
        assert title_count > 0, "Database should contain titles"

    print("2. Testing first get_personalized_picks call (should generate and cache)...")
    res1 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    assert "picks" in res1
    assert len(res1["picks"]) > 0
    picks1 = res1["picks"]
    pick_ids1 = [p["title_id"] for p in picks1]
    print(f"   Initial picks generated: {pick_ids1}")

    with get_db() as conn:
        cache_row = conn.execute("SELECT * FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert cache_row is not None, "Cache row for 'movie' should exist"
        cached_picks = json.loads(cache_row["picks_json"])
        assert len(cached_picks) == len(picks1)
        assert [p["title_id"] for p in cached_picks] == pick_ids1
        print("   [OK] Picks correctly persisted in for_you_cache table.")

    print("3. Testing second call (should load from cache, exact same picks)...")
    res2 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    picks2 = res2["picks"]
    pick_ids2 = [p["title_id"] for p in picks2]
    assert pick_ids1 == pick_ids2, f"Picks should match exactly: {pick_ids1} vs {pick_ids2}"
    print("   [OK] Second call returned identical cached picks without regenerating.")

    print("4. Testing dynamic watchlist rehydration on cached picks...")
    first_title_id = pick_ids1[0]
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist (title_id) VALUES (?)", (first_title_id,))

    res3 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    assert res3["picks"][0]["title_id"] == first_title_id
    assert res3["picks"][0]["is_on_watchlist"] is True, "First pick should now show is_on_watchlist=True"
    print("   [OK] Watchlist status rehydrated dynamically on cached picks.")

    print("5. Testing cache auto-invalidation on new rating...")
    with get_db() as conn:
        conn.execute("""
            INSERT INTO ratings (title_id, score, aspect_tags, notes)
            VALUES (?, 5, '["Visuals"]', 'Loved it')
            ON CONFLICT(title_id) DO UPDATE SET score=5
        """, (first_title_id,))

    res4 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    pick_ids4 = [p["title_id"] for p in res4["picks"]]
    assert first_title_id not in pick_ids4, f"Rated title {first_title_id} should be excluded on auto-refreshed picks: {pick_ids4}"
    print(f"   [OK] Rating status change automatically invalidated cache and regenerated picks: {pick_ids4}")

    print("6. Testing explicit force_refresh=True (should regenerate)...")
    res5 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=True)
    picks5 = res5["picks"]
    pick_ids5 = [p["title_id"] for p in picks5]
    # Since first_title_id was rated, it should be excluded from fresh recommendations
    assert first_title_id not in pick_ids5, f"Rated title {first_title_id} should be excluded on fresh refresh"
    print(f"   [OK] Force refresh generated fresh picks excluding rated title: {pick_ids5}")

    print("7. Testing independent caching for TV series vs Movies...")
    res_tv = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="tv", force_refresh=False)
    assert "picks" in res_tv
    with get_db() as conn:
        movie_cache = conn.execute("SELECT * FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        tv_cache = conn.execute("SELECT * FROM for_you_cache WHERE media_type = 'tv'").fetchone()
        assert movie_cache is not None, "Movie cache must exist"
        assert tv_cache is not None, "TV cache must exist"
        assert movie_cache["picks_json"] != tv_cache["picks_json"], "Movie and TV picks must be cached separately"
    print("   [OK] Movie and TV caches are maintained independently.")

    print("8. Testing FastAPI endpoints with TestClient...")
    client = TestClient(app)
    page_resp = client.get("/for-you")
    assert page_resp.status_code == 200
    assert "Made for your taste" in page_resp.text
    assert "Refresh picks" in page_resp.text

    api_resp = client.get("/api/for-you?media_type=movie&limit=5")
    assert api_resp.status_code == 200
    assert "for-you-results-target" in api_resp.text

    api_refresh_resp = client.get("/api/for-you?media_type=movie&limit=5&refresh=true")
    assert api_refresh_resp.status_code == 200
    assert "for-you-results-target" in api_refresh_resp.text
    print("   [OK] Endpoints /for-you and /api/for-you respond properly.")

    print("\nALL FOR-YOU CACHE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(test_for_you_cache())
