import asyncio
import json
from app.database import init_db, get_db
from app.services.orchestrator import orchestrator_service


async def test_why_reasons():
    print("1. Testing _generate_personalized_reason directly...")
    
    # Test 1: Loved title connection
    candidate = {
        "title": "The Great Gatsby",
        "genres": ["Drama", "Romance"],
        "director": "Baz Luhrmann",
        "imdb_rating": 7.3,
        "is_on_watchlist": False,
        "overview": "A writer and wall street trader finds himself drawn to the past and lifestyle of his millionaire neighbor."
    }
    loved_titles = [
        {
            "title": "La La Land",
            "genres": ["Comedy", "Drama", "Romance"],
            "score": 6,
            "director_or_creator": "Damien Chazelle"
        }
    ]
    liked_genres = {"Drama": 5, "Romance": 4}
    liked_creators = {"Baz Luhrmann": 1}
    
    reason_director = orchestrator_service._generate_personalized_reason(
        candidate, liked_creators=liked_creators
    )
    print(f"   Director reason: {reason_director}")
    assert "Baz Luhrmann" in reason_director
    assert "vector" not in reason_director.lower()
    
    reason_loved = orchestrator_service._generate_personalized_reason(
        candidate, loved_titles=loved_titles, liked_genres=liked_genres
    )
    print(f"   Loved title reason: {reason_loved}")
    assert "La La Land" in reason_loved
    assert "vector" not in reason_loved.lower()
    
    # Test 2: Watchlist reason
    candidate_wl = {**candidate, "is_on_watchlist": True}
    reason_wl = orchestrator_service._generate_personalized_reason(
        candidate_wl, liked_genres=liked_genres
    )
    print(f"   Watchlist reason: {reason_wl}")
    assert "Watchlist" in reason_wl
    assert "vector" not in reason_wl.lower()

    # Test 3: Heuristic personalized reasons
    heur_recs = orchestrator_service._heuristic_personalized_reasons(
        [candidate], liked_genres=liked_genres, liked_creators=liked_creators, loved_titles=loved_titles
    )
    print(f"   Heuristic reason: {heur_recs[0]['reason']}")
    assert "vector" not in heur_recs[0]['reason'].lower()
    assert "similarity" not in heur_recs[0]['reason'].lower()
    assert "openrouter" not in heur_recs[0]['reason'].lower()
    print("   [OK] Unit tests for reason generation passed.")

    print("\n2. Initializing DB and checking get_personalized_picks...")
    init_db()

    # Call get_personalized_picks
    res = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=True)
    picks = res["picks"]
    print(f"   Generated {len(picks)} picks.")
    assert len(picks) > 0, "Should generate picks"
    
    for i, p in enumerate(picks):
        title = p["title"]
        reason = p.get("reason", "")
        print(f"   Pick #{i+1}: '{title}' -> Reason: \"{reason}\"")
        assert reason, f"Pick '{title}' must have a non-empty reason"
        assert "vector" not in reason.lower(), f"Pick '{title}' contains vector jargon: {reason}"
        assert "openrouter" not in reason.lower(), f"Pick '{title}' contains API key instruction: {reason}"
        assert len(reason) >= 15, f"Pick '{title}' reason is too short: {reason}"

    print("\n3. Testing cache invalidation of stale vector text...")
    with get_db() as conn:
        # Inject old stale vector reason into cache
        fake_stale_picks = [
            {
                "title_id": picks[0]["title_id"],
                "title": picks[0]["title"],
                "reason": "High taste-vector similarity to your favorites."
            }
        ]
        conn.execute("""
            UPDATE for_you_cache
            SET picks_json = ?
            WHERE user_id = 'default_user' AND media_type = 'movie'
        """, (json.dumps(fake_stale_picks),))

    # Next call without force_refresh should detect stale vector text and auto-refresh!
    res_refreshed = await orchestrator_service.get_personalized_picks(limit=5, media_type_preference="movie", force_refresh=False)
    for p in res_refreshed["picks"]:
        assert "vector" not in p["reason"].lower(), f"Auto-refresh failed to clear vector jargon: {p['reason']}"
    print("   [OK] Cache auto-refresh successfully sanitized stale vector text.")

    print("\nALL REASON TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(test_why_reasons())
