"""Main user flows: pages, rating, Deck, For You, chat, dossier."""
from conftest import add_title, rate

from app.database import get_db
from app.services import chat, deck, for_you, taste_dossier
from app.services.ranking import matches_intent, normalize_intent


def test_pages_render(client):
    add_title(1, "Arrival")
    for url in ("/", "/chat", "/deck", "/deck?title_id=1", "/search", "/search?q=arr", "/library",
                "/library?tab=watchlist", "/library?tab=dossier", "/for-you", "/for-you?media_type=tv"):
        assert client.get(url).status_code == 200, url


def test_rating_watchlist_and_search(client):
    add_title(1, "Arrival")
    assert "Saved to Watchlist" in client.post("/api/watchlist/toggle/1").text
    assert client.post("/api/ratings", json={"title_id": 1, "score": 9}).status_code == 422
    assert client.post("/api/ratings", json={"title_id": 1, "score": 6, "notes": "Loved it"}).status_code == 200
    stats = client.get("/api/stats").text
    assert ">1</strong> rated" in stats and ">0</strong> saved" in stats  # rating clears the watchlist entry
    assert "Arrival" in client.get("/library").text
    assert "Arrival" in client.get("/api/search?q=arri").text
    with get_db() as conn:
        rating_id = conn.execute("SELECT id FROM ratings").fetchone()[0]
    client.delete(f"/api/ratings/{rating_id}")
    assert ">0</strong> rated" in client.get("/api/stats").text


def test_deck_serves_known_titles_and_recycles_skips(client):
    add_title(1, "Nobody Watched This", votes=7, rating=2.1)
    add_title(2, "Known Movie")
    assert "Known Movie" in client.get("/api/deck/next").text
    # Nothing fresh left: the oldest skipped quality title comes back, never the obscure one.
    card = client.post("/api/deck/skip/2").text
    assert "Known Movie" in card and "Nobody Watched This" not in card


async def test_deck_suggests_titles_like_loved_ones():
    add_title(1, "Loved A")
    add_title(2, "Loved B")
    rate(1, 6)
    rate(2, 6)
    add_title(3, "Similar")
    add_title(4, "Different", axis=5, genres=("Horror",))
    suggestion = await deck.get_deck_suggestion()
    assert suggestion["title"] == "Similar" and suggestion["is_suggestion"]


async def test_for_you_follows_ratings_caches_and_honours_skips(client):
    add_title(1, "Loved")
    rate(1, 6)
    add_title(2, "Similar")
    add_title(3, "Other", axis=7, genres=("Horror",))
    add_title(4, "A Series", media="tv")
    picks = (await for_you.get_personalized_picks(limit=2, media_type_preference="movie"))["picks"]
    assert [p["title"] for p in picks] == ["Similar", "Other"] and all(p["reason"] for p in picks)

    with get_db() as conn:
        conn.execute("UPDATE titles SET title = 'Renamed' WHERE id = 2")
    cached = (await for_you.get_personalized_picks(limit=2, media_type_preference="movie"))["picks"]
    assert cached[0]["title"] == "Similar"  # served from cache

    client.post("/api/for-you/skip/2?media_type=movie&limit=2")
    rate(3, 1)  # a new rating invalidates the cache
    picks = (await for_you.get_personalized_picks(limit=2, media_type_preference="movie"))["picks"]
    assert {p["title_id"] for p in picks}.isdisjoint({2, 3})


async def test_llm_picks_are_validated(fake_llm):
    add_title(1, "Loved")
    rate(1, 6)
    for title_id, name in ((2, "Good Pick"), (3, "Wrong Label"), (4, "Rejected")):
        add_title(title_id, name)
    fake_llm.replies["valid JSON for movie"] = {"intro": "Hi", "rejected_title_ids": [4], "picks": [
        {"title_id": 2, "title": "Good Pick", "reason": "A specific, well grounded reason."},
        {"title_id": 3, "title": "Not The Real Title", "reason": "A mislabelled pick is dropped."},
        {"title_id": 4, "title": "Rejected", "reason": "Rejected ids never appear."},
    ]}
    picks = (await for_you.get_personalized_picks(limit=3, media_type_preference="movie"))["picks"]
    # No backfill either: the model explicitly rejected the rest.
    assert [(p["title_id"], p["reason"]) for p in picks] == [(2, "A specific, well grounded reason.")]


async def test_chat_never_repeats_and_reloads_with_reasons(client):
    for i in range(1, 6):
        add_title(i, f"Space Film {i}", axis=i)
    first = await chat.handle_vibe_query("space adventure", session_id="s")
    second = await chat.handle_vibe_query("more like that", session_id="s")
    first_ids = {r["title_id"] for r in first["recommendations"]}
    assert first_ids and second["recommendations"]
    assert first_ids.isdisjoint(r["title_id"] for r in second["recommendations"])
    assert first["recommendations"][0]["reason"] in client.get("/chat?session_id=s").text


def test_intent_normalization_and_filters():
    intent = normalize_intent({"semantic_vibe": "awkward workplace comedy like The Office",
                               "reference_titles": ["The Office"], "genres": ["Sitcom", "Feel-Good"],
                               "excluded_genres": ["Horror"]}, "funny sitcom")
    assert intent["media_type"] == "tv" and intent["genres"] == ["Comedy"]
    assert "Office" not in intent["semantic_vibe"]
    assert matches_intent({"title": "Parks", "media_type": "tv", "genres": ["Comedy"]}, intent)
    assert not matches_intent({"title": "The Office", "media_type": "tv", "genres": ["Comedy"]}, intent)
    assert not matches_intent({"title": "Scary", "media_type": "tv", "genres": ["Comedy", "Horror"]}, intent)


async def test_dossier_needs_three_ratings_then_renders_llm_text_safely(client, fake_llm):
    for i in (1, 2, 3):
        add_title(i, f"Title {i}", axis=i)
    rate(1, 6)
    rate(2, 6)
    assert "at least 3 ratings" in (await taste_dossier.get_or_update_dossier())["full_summary"]
    rate(3, 2)
    fake_llm.replies["film taste"] = {"full_summary": "<b>bold</b> taste", "core_loves": ["Slow burns"]}
    html = client.post("/api/dossier/regenerate").text
    assert "&lt;b&gt;bold&lt;/b&gt;" in html and "Slow burns" in html
