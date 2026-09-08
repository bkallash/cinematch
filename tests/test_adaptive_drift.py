import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

import pytest
from starlette.testclient import TestClient

from app.config import settings
from app.database import init_db, get_db
from app.services.orchestrator import orchestrator_service
from app.services.taste_dossier import taste_dossier_service
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Ensure tests run deterministically with OpenRouter key unset."""
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    init_db()


@pytest.mark.asyncio
async def test_cache_invalidation_on_new_rating():
    """Verify that rating a currently recommended title invalidates the cache and excludes it."""
    with get_db() as conn:
        conn.execute("DELETE FROM for_you_cache")

    # Generate initial shelf
    res1 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    assert len(res1["picks"]) > 0
    first_pick_id = res1["picks"][0]["title_id"]
    cached_ids1 = [p["title_id"] for p in res1["picks"]]

    # Call again without modifications -> exact same picks served from cache
    res2 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    assert [p["title_id"] for p in res2["picks"]] == cached_ids1

    # Rate the first pick from the shelf
    with get_db() as conn:
        conn.execute("""
            INSERT INTO ratings (title_id, score, aspect_tags, notes, updated_at)
            VALUES (?, 6, '["Masterpiece"]', 'Instant classic', CURRENT_TIMESTAMP)
            ON CONFLICT(title_id) DO UPDATE SET score=6, updated_at=CURRENT_TIMESTAMP
        """, (first_pick_id,))

    # Next call without force_refresh must auto-refresh and exclude the newly rated title
    res3 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    refreshed_ids = [p["title_id"] for p in res3["picks"]]
    assert first_pick_id not in refreshed_ids
    assert refreshed_ids != cached_ids1


@pytest.mark.asyncio
async def test_cache_invalidation_on_rerating():
    """Verify that updating an existing rating's score and timestamp invalidates the cache."""
    with get_db() as conn:
        conn.execute("DELETE FROM for_you_cache")
        rating = conn.execute("SELECT id, title_id, score FROM ratings LIMIT 1").fetchone()
        assert rating is not None
        rid = rating["id"]

    # Generate and populate cache
    res1 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    with get_db() as conn:
        row1 = conn.execute("SELECT last_rated_at, ratings_count FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert row1 is not None

    # Re-rate: modify score and set future timestamp
    future_time = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute("""
            UPDATE ratings
            SET score = 1, updated_at = ?
            WHERE id = ?
        """, (future_time, rid))

    # Next call without force_refresh must detect the updated timestamp and auto-refresh
    res2 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    with get_db() as conn:
        row2 = conn.execute("SELECT last_rated_at FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert row2["last_rated_at"] == future_time


@pytest.mark.asyncio
async def test_cache_invalidation_on_rating_deletion():
    """Verify that deleting a rating invalidates the cache via ratings_count mismatch."""
    with get_db() as conn:
        conn.execute("DELETE FROM for_you_cache")
        rating = conn.execute("SELECT id, title_id FROM ratings LIMIT 1").fetchone()
        assert rating is not None
        rid = rating["id"]

    res1 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    with get_db() as conn:
        row1 = conn.execute("SELECT ratings_count FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        prev_count = row1["ratings_count"]

    # Delete rating
    with get_db() as conn:
        conn.execute("DELETE FROM ratings WHERE id = ?", (rid,))

    # Next call should auto-refresh and record decremented count in cache
    res2 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    with get_db() as conn:
        row2 = conn.execute("SELECT ratings_count FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert row2["ratings_count"] == prev_count - 1


@pytest.mark.asyncio
async def test_recency_and_dual_window_reactivity():
    """Verify dual-window reactivity: recent burst in new genre immediately shifts recommendations."""
    with get_db() as conn:
        conn.execute("DELETE FROM ratings")
        conn.execute("DELETE FROM for_you_cache")

        # Find 5 Drama and 3 Sci-Fi titles
        drama_titles = conn.execute("""
            SELECT id FROM titles
            WHERE genres LIKE '%Drama%' AND genres NOT LIKE '%Science Fiction%' AND embedding IS NOT NULL
            LIMIT 5
        """).fetchall()
        scifi_titles = conn.execute("""
            SELECT id FROM titles
            WHERE genres LIKE '%Science Fiction%' AND embedding IS NOT NULL
            LIMIT 3
        """).fetchall()

        assert len(drama_titles) >= 3
        assert len(scifi_titles) >= 2

        # 1. User historically rated Drama titles 360 days ago (~2 half-lives)
        old_time = (datetime.now(timezone.utc) - timedelta(days=360)).strftime("%Y-%m-%d %H:%M:%S")
        for t in drama_titles:
            conn.execute("""
                INSERT INTO ratings (title_id, score, aspect_tags, notes, created_at, updated_at)
                VALUES (?, 6, '[]', 'Old Drama love', ?, ?)
            """, (t["id"], old_time, old_time))

    # With only old Drama ratings, picks should be grounded in Drama
    res_initial = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=True)
    initial_genres = [g for p in res_initial["picks"] for g in p.get("genres", [])]
    assert "Drama" in initial_genres

    # 2. Taste drift: user logs a burst of fresh Science Fiction ratings today
    recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        for t in scifi_titles:
            conn.execute("""
                INSERT INTO ratings (title_id, score, aspect_tags, notes, created_at, updated_at)
                VALUES (?, 6, '[]', 'New SciFi obsession', ?, ?)
            """, (t["id"], recent_time, recent_time))

    # Next request auto-refreshes shelf; recent Sci-Fi burst must immediately surface in picks
    res_drift = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    drift_genres = [g for p in res_drift["picks"] for g in p.get("genres", [])]
    assert "Science Fiction" in drift_genres, f"Expected Science Fiction in drifted picks, got genres: {drift_genres}"


@pytest.mark.asyncio
async def test_deck_suggestion_alignment():
    """Verify that Rating Deck suggestions follow the user's recent ratings."""
    with get_db() as conn:
        conn.execute("DELETE FROM ratings")
        conn.execute("DELETE FROM for_you_cache")

        # Pick 3 Comedy titles to rate highly
        comedy_titles = conn.execute("""
            SELECT id FROM titles
            WHERE genres LIKE '%Comedy%' AND embedding IS NOT NULL
            LIMIT 3
        """).fetchall()
        assert len(comedy_titles) >= 2

        recent_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for t in comedy_titles:
            conn.execute("""
                INSERT INTO ratings (title_id, score, aspect_tags, notes, created_at, updated_at)
                VALUES (?, 6, '[]', 'Loved comedy', ?, ?)
            """, (t["id"], recent_time, recent_time))

    suggestion = await orchestrator_service.get_deck_suggestion()
    assert suggestion is not None, "Expected personalized deck suggestion for user with 3 ratings"
    assert suggestion["is_suggestion"] is True
    assert 75 <= suggestion["match_score"] <= 98
    assert (
        "Matches your" in suggestion["suggestion_reason"]
        or "Tailored to your" in suggestion["suggestion_reason"]
    )


@pytest.mark.asyncio
async def test_taste_dossier_resynthesis_and_age_annotation():
    """Verify Taste Dossier re-synthesizes on rating changes and formats age in prompt."""
    with get_db() as conn:
        conn.execute("DELETE FROM ratings")
        conn.execute("DELETE FROM taste_dossiers")

        sample_titles = conn.execute("SELECT id FROM titles LIMIT 3").fetchall()
        t1, t2, t3 = sample_titles[0][0], sample_titles[1][0], sample_titles[2][0]

        now = datetime.now(timezone.utc)
        ts_today = now.strftime("%Y-%m-%d %H:%M:%S")
        ts_3days = (now - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
        ts_60days = (now - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")

        conn.execute("INSERT INTO ratings (title_id, score, created_at, updated_at) VALUES (?, 6, ?, ?)", (t1, ts_today, ts_today))
        conn.execute("INSERT INTO ratings (title_id, score, created_at, updated_at) VALUES (?, 5, ?, ?)", (t2, ts_3days, ts_3days))
        conn.execute("INSERT INTO ratings (title_id, score, created_at, updated_at) VALUES (?, 4, ?, ?)", (t3, ts_60days, ts_60days))

    # Without API key, fallback returns placeholder summary with exact count
    dossier = await taste_dossier_service.get_or_update_dossier(force=True)
    assert dossier["ratings_count_at_synthesis"] == 3

    # Add 4th rating and mark dirty
    with get_db() as conn:
        t4 = conn.execute("SELECT id FROM titles WHERE id NOT IN (?, ?, ?) LIMIT 1", (t1, t2, t3)).fetchone()[0]
        conn.execute("INSERT INTO ratings (title_id, score, created_at, updated_at) VALUES (?, 6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", (t4,))
    taste_dossier_service.mark_dirty()

    dossier2 = await taste_dossier_service.get_or_update_dossier()
    assert dossier2["ratings_count_at_synthesis"] == 4

    # Verify age formatting helper outputs
    assert taste_dossier_service._format_age(ts_today) == "rated today"
    assert taste_dossier_service._format_age(ts_3days) == "rated 3 days ago"
    assert "month" in taste_dossier_service._format_age(ts_60days)


def test_http_api_auto_refresh_on_rating():
    """Verify via TestClient that POST /api/ratings auto-refreshes GET /api/for-you."""
    client = TestClient(app)

    # Initial fetch: populates cache
    resp1 = client.get("/api/for-you?media_type=movie&limit=5")
    assert resp1.status_code == 200

    # Rate an unrated title via HTTP API
    with get_db() as conn:
        unrated = conn.execute("""
            SELECT id FROM titles
            WHERE id NOT IN (SELECT title_id FROM ratings)
            LIMIT 1
        """).fetchone()
        assert unrated is not None
        title_id = unrated[0]

    rate_resp = client.post("/api/ratings", json={
        "title_id": title_id,
        "score": 5,
        "aspect_tags": ["Visuals"],
        "notes": "Testing auto-refresh via HTTP"
    })
    assert rate_resp.status_code == 200

    # Next GET /api/for-you without refresh=true must return 200 and auto-refreshed content
    resp2 = client.get("/api/for-you?media_type=movie&limit=5")
    assert resp2.status_code == 200
    assert f'data-title-id="{title_id}"' not in resp2.text


async def main():
    print("Running adaptive taste drift tests...")
    settings.OPENROUTER_API_KEY = ""
    init_db()

    await test_cache_invalidation_on_new_rating()
    print("  [OK] test_cache_invalidation_on_new_rating")

    await test_cache_invalidation_on_rerating()
    print("  [OK] test_cache_invalidation_on_rerating")

    await test_cache_invalidation_on_rating_deletion()
    print("  [OK] test_cache_invalidation_on_rating_deletion")

    await test_recency_and_dual_window_reactivity()
    print("  [OK] test_recency_and_dual_window_reactivity")

    await test_deck_suggestion_alignment()
    print("  [OK] test_deck_suggestion_alignment")

    await test_taste_dossier_resynthesis_and_age_annotation()
    print("  [OK] test_taste_dossier_resynthesis_and_age_annotation")

    test_http_api_auto_refresh_on_rating()
    print("  [OK] test_http_api_auto_refresh_on_rating")

    print("\nALL ADAPTIVE TASTE DRIFT TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
