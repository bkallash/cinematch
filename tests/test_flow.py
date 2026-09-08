import asyncio
import os
import json
from app.database import init_db, get_db
from app.services.seeder import seed_starter_catalog
from app.services.embeddings import embedding_service
from app.services.orchestrator import orchestrator_service
from app.services.taste_dossier import taste_dossier_service

async def test_full_pipeline():
    print("1. Initializing database...")
    init_db()

    print("2. Seeding starter catalog...")
    await seed_starter_catalog()

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
        with_embed = conn.execute("SELECT COUNT(*) FROM titles WHERE embedding IS NOT NULL").fetchone()[0]
        print(f"   Database contains {count} titles ({with_embed} with vector embeddings).")
        assert count > 0, "Catalog should have titles"
        assert with_embed > 0, "Titles should have vector embeddings"

        first_title = conn.execute("SELECT id, title, media_type, release_year FROM titles LIMIT 1").fetchone()
        title_id = first_title["id"]
        print(f"   Found title #{title_id}: '{first_title['title']}' ({first_title['release_year']}, {first_title['media_type']})")

    print("3. Testing Rating submission (1-6 scale with aspect tags)...")
    with get_db() as conn:
        conn.execute("""
            INSERT INTO ratings (title_id, score, aspect_tags, notes)
            VALUES (?, 6, ?, 'Absolute masterpiece!')
            ON CONFLICT(title_id) DO UPDATE SET score=6, aspect_tags=excluded.aspect_tags
        """, (title_id, json.dumps(["Visuals", "Soundtrack", "Ending"])))
    taste_dossier_service.mark_dirty()

    with get_db() as conn:
        rating = conn.execute("SELECT * FROM ratings WHERE title_id = ?", (title_id,)).fetchone()
        assert rating["score"] == 6
        assert "Visuals" in rating["aspect_tags"]
        print(f"   Successfully logged rating: {rating['score']}/6 with tags {rating['aspect_tags']}")

    print("4. Testing Watchlist addition...")
    with get_db() as conn:
        second_title = conn.execute("SELECT id, title FROM titles WHERE id != ? LIMIT 1", (title_id,)).fetchone()
        second_id = second_title["id"]
        conn.execute("INSERT OR IGNORE INTO watchlist (title_id) VALUES (?)", (second_id,))
        wl_row = conn.execute("SELECT * FROM watchlist WHERE title_id = ?", (second_id,)).fetchone()
        assert wl_row is not None
        print(f"   Successfully added '{second_title['title']}' to Watchlist.")

    print("5. Testing Orchestrator vector candidate search...")
    intent = {
        "semantic_vibe": "mind-bending space travel, wormholes, physics and emotional father-daughter bond",
        "media_type": None,
        "genres": ["Science Fiction"],
        "year_min": None,
        "year_max": None
    }
    candidates = await orchestrator_service._search_local_candidates(intent)
    print(f"   Found {len(candidates)} candidate matches.")
    if candidates:
        top = candidates[0]
        print(f"   Top Match: '{top['title']}' (Score: {top['similarity']:.3f})")

    print("6. Testing Taste Dossier status...")
    dossier = await taste_dossier_service.get_or_update_dossier()
    print(f"   Dossier summary: {dossier['full_summary'][:120]}...")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
