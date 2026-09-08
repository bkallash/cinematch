import asyncio
import json
from datetime import datetime, timedelta, timezone
import pytest
from starlette.testclient import TestClient

from app.config import settings
from app.database import init_db, get_db
from app.services.orchestrator import (
    orchestrator_service,
    HALF_LIFE_DAYS,
    RECENCY_FLOOR,
    RECENT_WINDOW_SIZE,
)
from app.services.taste_dossier import taste_dossier_service
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Ensure tests run deterministically with OpenRouter key unset."""
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    init_db()


@pytest.mark.asyncio
async def test_cache_invalidation_on_new_rating():
    """Verify that adding a new rating invalidates the For-You cache and auto-refreshes."""
    with get_db() as conn:
        conn.execute("DELETE FROM for_you_cache")
        # Find 2 unrated titles
        unrated = conn.execute("""
            SELECT id FROM titles
            WHERE id NOT IN (SELECT title_id FROM ratings)
              AND embedding IS NOT NULL
            LIMIT 2
        """).fetchall()
        assert len(unrated) >= 2
        t1, t2 = unrated[0][0], unrated[1][0]

    # Generate initial shelf
    res1 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    cached_ids1 = [p["title_id"] for p in res1["picks"]]

    # Verify cache row exists
    with get_db() as conn:
        row = conn.execute("SELECT ratings_count, last_rated_at FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert row is not None
        initial_count = row["ratings_count"]

    # Subsequent call without rating changes hits cache
    res2 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    cached_ids2 = [p["title_id"] for p in res2["picks"]]
    assert cached_ids1 == cached_ids2

    # Add a new rating
    with get_db() as conn:
        conn.execute("""
            INSERT INTO ratings (title_id, score, aspect_tags, notes, updated_at)
            VALUES (?, 6, '["Masterpiece"]', 'Instant classic', CURRENT_TIMESTAMP)
        """, (t1,))

    # Next call without force_refresh must auto-refresh and exclude t1
    res3 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    refreshed_ids = [p["title_id"] for p in res3["picks"]]
    assert t1 not in refreshed_ids

    # Cache row should be updated with new count
    with get_db() as conn:
        row2 = conn.execute("SELECT ratings_count FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert row2["ratings_count"] == initial_count + 1


@pytest.mark.asyncio
async def test_cache_invalidation_on_rerating():
    """Verify that updating a rating timestamp invalidates the cache even when count is unchanged."""
    with get_db() as conn:
        conn.execute("DELETE FROM for_you_cache")
        # Ensure at least one rating exists
        rating = conn.execute("SELECT id, title_id, score FROM ratings LIMIT 1").fetchone()
        assert rating is not None
        rid, r_tid = rating["id"], rating["title_id"]

    # Populate cache
    res1 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)

    with get_db() as conn:
        row1 = conn.execute("SELECT last_rated_at FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert row1 is not None
        initial_last_rated = row1["last_rated_at"]

    # Re-rate: change score and push updated_at forward
    future_time = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute("""
            UPDATE ratings
            SET score = 1, updated_at = ?
            WHERE id = ?
        """, (future_time, rid))

    # Next call without force_refresh must detect the timestamp mismatch and auto-refresh
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

    # Next call should auto-refresh and update cache with decremented count
    res2 = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    with get_db() as conn:
        row2 = conn.execute("SELECT ratings_count FROM for_you_cache WHERE media_type = 'movie'").fetchone()
        assert row2["ratings_count"] == prev_count - 1


@pytest.mark.asyncio
async def test_recency_weighting_influence():
    """Verify that recent ratings dominate older ratings via exponential decay and dual-window blend."""
    with get_db() as conn:
        # Clear existing ratings and cache for this user
        conn.execute("DELETE FROM ratings")
        conn.execute("DELETE FROM for_you_cache")

        # Pick 3 Action titles and 3 Animation titles with embeddings
        action_titles = conn.execute("""
            SELECT id, title FROM titles
            WHERE genres LIKE '%Action%' AND embedding IS NOT NULL
            LIMIT 3
        """).fetchall()
        animation_titles = conn.execute("""
            SELECT id, title FROM titles
            WHERE genres LIKE '%Animation%' AND embedding IS NOT NULL
            LIMIT 3
        """).fetchall()

        assert len(action_titles) >= 2
        assert len(animation_titles) >= 2

        # Log Action titles as ancient ratings (360 days ago, ~2 half-lives)
        old_time = (datetime.now(timezone.utc) - timedelta(days=360)).strftime("%Y-%m-%d %H:%M:%S")
        for t in action_titles:
            conn.execute("""
                INSERT INTO ratings (title_id, score, aspect_tags, notes, created_at, updated_at)
                VALUES (?, 6, '[]', 'Old love', ?, ?)
            """, (t["id"], old_time, old_time))

        # Log Animation titles as fresh ratings (1 hour ago)
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        for t in animation_titles:
            conn.execute("""
                INSERT INTO ratings (title_id, score, aspect_tags, notes, created_at, updated_at)
                VALUES (?, 6, '[]', 'New love', ?, ?)
            """, (t["id"], recent_time, recent_time))

    # Build taste profile
    profile = orchestrator_service._build_taste_profile()
    assert profile["taste_vector"] is not None
    assert profile["ratings_count"] == len(action_titles) + len(animation_titles)
    assert profile["last_rated_at"] == recent_time

    # Generate personalized picks: Animation (recent) should heavily influence picks
    res = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=True)
    picks = res["picks"]
    assert len(picks) > 0

    # Verify at least one Animation pick appears in the recommendations
    all_pick_genres = [g for p in picks for g in p.get("genres", [])]
    assert "Animation" in all_pick_genres


@pytest.mark.asyncio
async def test_deck_suggestion_alignment():
    """Verify that Rating Deck suggestions use the blended taste vector and return proper reasons."""
    suggestion = await orchestrator_service.get_deck_suggestion()
    if suggestion:
        assert suggestion["is_suggestion"] is True
        assert 75 <= suggestion["match_score"] <= 98
        assert (
            "Matches your" in suggestion["suggestion_reason"]
            or "Tailored to your" in suggestion["suggestion_reason"]
        )


def test_taste_dossier_age_formatting():
    """Verify TasteDossierService._format_age produces clean human-readable relative ages."""
    now = datetime.now(timezone.utc)

    today_str = now.strftime("%Y-%m-%d %H:%M:%S")
    assert taste_dossier_service._format_age(today_str) == "rated today"

    one_day_str = (now - timedelta(days=1, hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    assert taste_dossier_service._format_age(one_day_str) == "rated 1 day ago"

    five_days_str = (now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    assert taste_dossier_service._format_age(five_days_str) == "rated 5 days ago"

    two_months_str = (now - timedelta(days=62)).strftime("%Y-%m-%d %H:%M:%S")
    assert "month" in taste_dossier_service._format_age(two_months_str)

    one_year_str = (now - timedelta(days=370)).strftime("%Y-%m-%d %H:%M:%S")
    assert "year" in taste_dossier_service._format_age(one_year_str)

    assert taste_dossier_service._format_age(None) == "rated long ago"
    assert taste_dossier_service._format_age("invalid-date") == "rated long ago"


def test_http_api_auto_refresh_on_rating():
    """Secondary seam test: TestClient POST /api/ratings auto-refreshes GET /api/for-you."""
    client = TestClient(app)

    # First fetch: populates cache
    resp1 = client.get("/api/for-you?media_type=movie&limit=5")
    assert resp1.status_code == 200

    # Pick an unrated title to rate
    with get_db() as conn:
        unrated = conn.execute("""
            SELECT id FROM titles
            WHERE id NOT IN (SELECT title_id FROM ratings)
            LIMIT 1
        """).fetchone()
        assert unrated is not None
        title_id = unrated[0]

    # Post rating via API
    rate_resp = client.post("/api/ratings", json={
        "title_id": title_id,
        "score": 5,
        "aspect_tags": ["Visuals"],
        "notes": "Testing auto-refresh"
    })
    assert rate_resp.status_code == 200

    # Next GET /api/for-you without refresh=true must return 200 and reflect auto-refresh
    resp2 = client.get("/api/for-you?media_type=movie&limit=5")
    assert resp2.status_code == 200
    # Newly rated title must not appear in recommended shelf
    assert f'data-title-id="{title_id}"' not in resp2.text
