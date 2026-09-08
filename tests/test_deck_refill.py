import asyncio
import json
from app.database import get_db, init_db
from app.main import (
    get_next_deck_title,
    refill_deck_with_preferences_and_famous,
    schedule_deck_refill_if_needed,
    format_title_row,
    DECK_SUGGESTION_CADENCE,
)
from app.services.orchestrator import orchestrator_service


async def test_deck_returns_title():
    """Verify get_next_deck_title returns a formatted title quickly."""
    init_db()
    title = await get_next_deck_title()
    assert title is not None, "Deck should return a title"
    assert "id" in title
    assert "title" in title
    assert "media_type" in title
    assert "genres" in title
    assert isinstance(title["genres"], list)
    print(f"   [OK] Deck returned title #{title['id']}: '{title['title']}' ({title['media_type']})")


async def test_deck_suggestion_cadence_and_metadata():
    """Verify that every Nth card is evaluated for a personalized suggestion."""
    init_db()

    # Get a personalized suggestion directly to check structure
    suggestion = await orchestrator_service.get_deck_suggestion()
    if suggestion:
        assert suggestion["is_suggestion"] is True
        assert suggestion["match_score"] is not None
        assert isinstance(suggestion["match_score"], int)
        assert suggestion["suggestion_reason"] is not None
        assert "Matches your" in suggestion["suggestion_reason"] or "Tailored to your" in suggestion["suggestion_reason"]

    # Verify multiple calls to get_next_deck_title keep yielding cards
    cards = []
    for _ in range(DECK_SUGGESTION_CADENCE * 2):
        card = await get_next_deck_title()
        assert card is not None
        cards.append(card)

    assert len(cards) == DECK_SUGGESTION_CADENCE * 2
    # At least one card in the cadence sequence should have suggestion metadata if user has ratings
    with get_db() as conn:
        ratings_cnt = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    if ratings_cnt >= 3:
        has_suggestion = any(c.get("is_suggestion") for c in cards)
        assert has_suggestion, "Expected at least one personalized suggestion across cadence window"


async def test_format_title_row_with_suggestion():
    """Verify format_title_row properly includes suggestion metadata."""
    sample_data = {
        "id": 999,
        "tmdb_id": 12345,
        "media_type": "movie",
        "title": "Test Cinema",
        "original_title": "Test Cinema",
        "release_year": 2024,
        "overview": "A test movie overview.",
        "poster_path": "/test.jpg",
        "backdrop_path": "/test_bg.jpg",
        "genres": ["Drama", "Sci-Fi"],
        "director_or_creator": "Test Director",
        "cast_top": ["Actor A", "Actor B"],
        "vote_average": 8.5,
        "vote_count": 2500,
        "popularity": 120.0,
        "is_suggestion": True,
        "suggestion_reason": "Matches your taste in Sci-Fi",
        "match_score": 93,
    }
    formatted = format_title_row(sample_data)
    assert formatted["is_suggestion"] is True
    assert formatted["suggestion_reason"] == "Matches your taste in Sci-Fi"
    assert formatted["match_score"] == 93
    print("   [OK] format_title_row preserves suggestion metadata")


async def test_refill_mechanism():
    """Verify preference-and-famous refill runs without error."""
    inserted = await refill_deck_with_preferences_and_famous()
    assert isinstance(inserted, int)
    assert inserted >= 0
    print(f"   [OK] refill_deck_with_preferences_and_famous cached {inserted} titles")


if __name__ == "__main__":
    asyncio.run(test_deck_returns_title())
    asyncio.run(test_deck_suggestion_cadence_and_metadata())
    asyncio.run(test_format_title_row_with_suggestion())
    asyncio.run(test_refill_mechanism())
    print("\nAll deck refill & suggestion tests passed!")
