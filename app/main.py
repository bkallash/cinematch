import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.database import get_db, init_db
from app.models.schemas import RatingCreate
from app.services.catalog import ensure_title_embedding, popular_titles, search_catalog
from app.services.deck import get_deck_title_by_id, get_next_deck_title
from app.services.chat import handle_vibe_query
from app.services.for_you import get_personalized_picks
from app.services.seeder import seed_starter_catalog
from app.services.taste_dossier import get_or_update_dossier
from app.services.tmdb import tmdb_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cinematch")

BASE_DIR = Path(__file__).resolve().parent
_background_tasks: set = set()  # the event loop keeps only weak references to tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        await seed_starter_catalog()
    except Exception as e:
        logger.error(f"Error running starter catalog seeder: {e}")
    yield
    await tmdb_service.aclose()


app = FastAPI(title="CineMatch", description="AI-Powered Movie & Series Suggestion App", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def filter_backdrop_url(backdrop_path: Optional[str], fallback_poster: Optional[str] = None) -> str:
    if not backdrop_path:
        return tmdb_service.get_poster_url(fallback_poster)
    return backdrop_path if backdrop_path.startswith("http") else f"{settings.TMDB_IMAGE_BASE_URL}{backdrop_path}"


templates.env.filters["poster_url"] = tmdb_service.get_poster_url
templates.env.filters["backdrop_url"] = filter_backdrop_url


def get_global_stats() -> dict:
    with get_db() as conn:
        return {
            "ratings_count": conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0],
            "watchlist_count": conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0],
        }


def render_page(request: Request, name: str, active_tab: str, **context) -> HTMLResponse:
    """Render a full page with the navbar stats every page shows."""
    return templates.TemplateResponse(request, name, {"active_tab": active_tab, **get_global_stats(), **context})


def render_component(request: Request, name: str, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, f"components/{name}", context)


def normalize_media_type(media_type: str) -> str:
    return media_type if media_type in ("movie", "tv") else "movie"


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def load_chat_messages(session_id: str) -> list:
    """Rebuild a saved chat with its recommendation cards."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT role, content, recommended_title_ids, recommendation_reasons FROM chat_messages
            WHERE session_id = ? ORDER BY created_at ASC
        """, (session_id,)).fetchall()
        rec_ids = [json.loads(r["recommended_title_ids"] or "[]") for r in rows]
        all_ids = sorted({tid for ids in rec_ids for tid in ids})
        titles = {t["id"]: t for t in conn.execute(f"""
            SELECT t.*, w.id IS NOT NULL AS is_on_watchlist
            FROM titles t LEFT JOIN watchlist w ON t.id = w.title_id
            WHERE t.id IN ({",".join("?" * len(all_ids))})
        """, all_ids)}

    def card(t, reasons):
        # Messages saved before reasons were stored fall back to a generic line.
        return {
            "title_id": t["id"], "tmdb_id": t["tmdb_id"], "media_type": t["media_type"],
            "title": t["title"], "release_year": t["release_year"], "poster_path": t["poster_path"],
            "genres": json.loads(t["genres"] or "[]"), "overview": t["overview"],
            "reason": reasons.get(str(t["id"]), "Recommended earlier in this chat."),
            "is_on_watchlist": bool(t["is_on_watchlist"]), "vote_average": t["vote_average"],
            "imdb_rating": t["imdb_rating"] or t["vote_average"], "imdb_id": t["imdb_id"],
        }

    return [{"role": r["role"], "content": r["content"],
             "recommendations": [card(titles[tid], json.loads(r["recommendation_reasons"] or "{}"))
                                 for tid in ids if tid in titles]}
            for r, ids in zip(rows, rec_ids)]


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def page_assistant(request: Request, session_id: Optional[str] = None):
    # Without an explicit session, every page load starts a fresh chat.
    return render_page(request, "chat.html", "chat",
                       messages=load_chat_messages(session_id) if session_id else [],
                       session_id=session_id or f"session_{uuid.uuid4().hex[:12]}")


@app.get("/deck", response_class=HTMLResponse)
async def page_deck(request: Request, title_id: Optional[int] = None):
    title = await get_deck_title_by_id(title_id) if title_id is not None else None
    return render_page(request, "deck.html", "deck", title=title or await get_next_deck_title())


@app.get("/search", response_class=HTMLResponse)
async def page_search(request: Request, q: Optional[str] = None):
    query = (q or "").strip()
    results = await search_catalog(query) if query else popular_titles()
    return render_page(request, "search.html", "search", query=q or "", results=results)


@app.get("/library", response_class=HTMLResponse)
async def page_library(request: Request, tab: str = "ratings"):
    ratings, watchlist, dossier = [], [], {}
    if tab == "dossier":
        dossier = await get_or_update_dossier()
    elif tab == "watchlist":
        with get_db() as conn:
            rows = conn.execute("""
                SELECT w.id, w.notes, t.id AS title_id, t.title, t.media_type, t.release_year,
                       t.poster_path, t.overview, t.genres
                FROM watchlist w JOIN titles t ON w.title_id = t.id
                ORDER BY w.created_at DESC
            """).fetchall()
        watchlist = [{**dict(r), "genres": json.loads(r["genres"] or "[]")} for r in rows]
    else:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT r.id, r.score, r.aspect_tags, r.notes, t.id AS title_id, t.title, t.media_type,
                       t.release_year, t.poster_path, t.genres, t.director_or_creator AS director
                FROM ratings r JOIN titles t ON r.title_id = t.id
                ORDER BY r.created_at DESC
            """).fetchall()
        ratings = [{**dict(r), "aspect_tags": json.loads(r["aspect_tags"] or "[]"),
                    "genres": json.loads(r["genres"] or "[]")} for r in rows]

    return render_page(request, "library.html", "library", current_tab=tab,
                       ratings=ratings, watchlist=watchlist, dossier=dossier)


@app.get("/for-you", response_class=HTMLResponse)
async def page_for_you(request: Request, media_type: str = "movie", refresh: bool = False):
    """Personal shelf: the 5 Movies or Series most likely to be liked."""
    media_type = normalize_media_type(media_type)
    result = await get_personalized_picks(
        limit=5, media_type_preference=media_type, force_refresh=refresh)
    return render_page(request, "for_you.html", "for-you", media_type=media_type, **result)


# --------------------------------------------------------------------------
# HTMX & API endpoints
# --------------------------------------------------------------------------

@app.get("/api/for-you", response_class=HTMLResponse)
async def api_for_you(request: Request, media_type: str = "movie", limit: int = 5,
                      refresh: bool = False, format: Optional[str] = None):
    """HTMX partial (or JSON with ?format=json) for the personalized shelf."""
    media_type = normalize_media_type(media_type)
    result = await get_personalized_picks(
        limit=max(1, min(limit, 10)), media_type_preference=media_type, force_refresh=refresh)
    if format == "json" or "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content=result)
    return render_component(request, "for_you_results.html", media_type=media_type, is_htmx=True, **result)


@app.post("/api/for-you/skip/{title_id}", response_class=HTMLResponse)
async def api_for_you_skip(request: Request, title_id: int, media_type: str = "movie", limit: int = 5):
    """Dismiss a Title from future For You shelves and return replacement picks."""
    with get_db() as conn:
        if not conn.execute("SELECT id FROM titles WHERE id = ?", (title_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Title not found")
        conn.execute("INSERT OR IGNORE INTO for_you_skips (title_id) VALUES (?)", (title_id,))
    return await api_for_you(request, media_type=media_type, limit=limit)


@app.post("/api/chat", response_class=HTMLResponse)
async def api_chat(message: str = Form(...), session_id: str = Form("default_session"),
                   media_type: str = Form("all")):
    """Answer a chat request and return the rendered user + assistant messages."""
    result = await handle_vibe_query(
        query_text=message, session_id=session_id, media_type_preference=media_type)
    template = templates.get_template("components/chat_message.html")
    return HTMLResponse(
        template.render(role="user", content=message)
        + template.render(role="assistant", content=result["assistant_message"],
                          recommendations=result["recommendations"]))


@app.post("/api/chat/clear", response_class=HTMLResponse)
async def api_chat_clear(session_id: str = Form("default_session")):
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    return HTMLResponse("")


async def next_deck_card(request: Request) -> HTMLResponse:
    return render_component(request, "deck_card.html", title=await get_next_deck_title())


@app.get("/api/deck/next", response_class=HTMLResponse)
async def api_deck_next(request: Request):
    return await next_deck_card(request)


@app.post("/api/deck/skip/{title_id}", response_class=HTMLResponse)
async def api_deck_skip(request: Request, title_id: int):
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO skipped_titles (title_id) VALUES (?)", (title_id,))
    return await next_deck_card(request)


@app.post("/api/deck/watchlist/{title_id}", response_class=HTMLResponse)
async def api_deck_watchlist(request: Request, title_id: int):
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist (title_id) VALUES (?)", (title_id,))
        # Also mark it skipped so the Deck moves past it.
        conn.execute("INSERT OR IGNORE INTO skipped_titles (title_id) VALUES (?)", (title_id,))
    return await next_deck_card(request)


@app.post("/api/ratings")
async def api_save_rating(payload: RatingCreate):
    # The ratings triggers mark the dossier dirty and clear the For You cache.
    with get_db() as conn:
        conn.execute("""
            INSERT INTO ratings (title_id, score, aspect_tags, notes, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(title_id) DO UPDATE SET
                score = excluded.score,
                aspect_tags = excluded.aspect_tags,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
        """, (payload.title_id, payload.score, json.dumps(payload.aspect_tags), payload.notes))
        conn.execute("DELETE FROM watchlist WHERE title_id = ?", (payload.title_id,))
        conn.execute("DELETE FROM skipped_titles WHERE title_id = ?", (payload.title_id,))

    # Rated titles need full details and an embedding to inform recommendations.
    task = asyncio.create_task(ensure_title_embedding(payload.title_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"status": "ok", "title_id": payload.title_id, "score": payload.score}


@app.delete("/api/ratings/{rating_id}", response_class=HTMLResponse)
async def api_delete_rating(rating_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM ratings WHERE id = ?", (rating_id,))
    return HTMLResponse("")


@app.post("/api/watchlist/toggle/{title_id}", response_class=HTMLResponse)
async def api_toggle_watchlist(request: Request, title_id: int):
    with get_db() as conn:
        removed = conn.execute("DELETE FROM watchlist WHERE title_id = ?", (title_id,)).rowcount
        if not removed:
            conn.execute("INSERT INTO watchlist (title_id) VALUES (?)", (title_id,))
    return render_component(request, "watchlist_button.html", title_id=title_id, in_watchlist=not removed)


@app.delete("/api/watchlist/{watchlist_id}", response_class=HTMLResponse)
async def api_delete_watchlist(watchlist_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM watchlist WHERE id = ?", (watchlist_id,))
    return HTMLResponse("")


@app.get("/api/search", response_class=HTMLResponse)
async def api_search(request: Request, q: str = ""):
    return render_component(request, "search_results.html", query=q, results=await search_catalog(q.strip()))


@app.get("/api/stats", response_class=HTMLResponse)
async def api_stats(request: Request):
    return render_component(request, "nav_stats.html", **get_global_stats())


@app.post("/api/dossier/regenerate", response_class=HTMLResponse)
async def api_dossier_regenerate(request: Request):
    dossier = await get_or_update_dossier(force=True)
    return render_component(request, "dossier.html", dossier=dossier)
