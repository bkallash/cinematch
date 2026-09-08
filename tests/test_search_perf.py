import asyncio
import time
from app.database import init_db, get_db
from app.main import perform_title_search, get_deck_title_by_id

async def test_search_speed_and_completeness():
    init_db()

    # Search for a query that requires TMDB fallback
    query = "Arrival"
    start_time = time.perf_counter()
    results = await perform_title_search(query)
    elapsed = time.perf_counter() - start_time

    print(f"Search for '{query}' returned {len(results)} results in {elapsed:.2f}s")
    assert elapsed < 2.0, f"Search took too long: {elapsed:.2f}s (expected < 2.0s)"
    assert len(results) > 0, "Expected at least one search result"

    first = results[0]
    assert "id" in first and first["id"] > 0, "Result should have a valid SQLite database ID"
    assert "title" in first and first["title"], "Result should have a title"
    assert "media_type" in first, "Result should have media_type"

    # Verify that opening in Rating Deck handles the title properly (and fetches on-demand details if needed)
    deck_title = await get_deck_title_by_id(first["id"])
    assert deck_title is not None, "Deck should load title by id"
    assert deck_title["id"] == first["id"]

    print(f"[OK] Search performance and deck retrieval test passed in {elapsed:.2f}s!")

if __name__ == "__main__":
    asyncio.run(test_search_speed_and_completeness())
