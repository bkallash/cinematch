import asyncio
import json
import logging
import uuid
from html import escape
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, Query, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import settings
from app.database import get_db, init_db
from app.services.tmdb import tmdb_service
from app.services.embeddings import embedding_service
from app.services.taste_dossier import taste_dossier_service
from app.services.orchestrator import (
    DECK_MIN_VOTE_AVERAGE,
    DECK_MIN_VOTE_COUNT,
    orchestrator_service,
)
from app.services.seeder import seed_starter_catalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cinematch")

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    init_db()
    # Seed starter catalog
    try:
        await seed_starter_catalog()
    except Exception as e:
        logger.error(f"Error running starter catalog seeder: {e}")
    yield

app = FastAPI(
    title="CineMatch",
    description="AI-Powered Movie & Series Suggestion App",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Jinja2 Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Custom Jinja Filters
def filter_poster_url(poster_path: Optional[str]) -> str:
    return tmdb_service.get_poster_url(poster_path)

def filter_backdrop_url(backdrop_path: Optional[str], fallback_poster: Optional[str] = None) -> str:
    if backdrop_path:
        if backdrop_path.startswith("http"):
            return backdrop_path
        return f"{settings.TMDB_IMAGE_BASE_URL}{backdrop_path}"
    return filter_poster_url(fallback_poster)

templates.env.filters["poster_url"] = filter_poster_url
templates.env.filters["backdrop_url"] = filter_backdrop_url

# Helper to get global navbar stats
def get_global_stats() -> dict:
    with get_db() as conn:
        ratings_cnt = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
        watchlist_cnt = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
    return {"ratings_count": ratings_cnt, "watchlist_count": watchlist_cnt}


# --------------------------------------------------------------------------
# Web UI Pages
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def page_assistant(request: Request, session_id: Optional[str] = None):
    stats = get_global_stats()

    # If an explicit session_id query param is supplied, load its messages
    if session_id:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT role, content, recommended_title_ids
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()

        messages = []
        for r in rows:
            recs = []
            rec_ids = json.loads(r["recommended_title_ids"] or "[]")
            if rec_ids:
                placeholders = ",".join("?" * len(rec_ids))
                with get_db() as conn:
                    rec_rows = conn.execute(f"""
                        SELECT t.*, CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_on_watchlist
                        FROM titles t
                        LEFT JOIN watchlist w ON t.id = w.title_id
                        WHERE t.id IN ({placeholders})
                    """, rec_ids).fetchall()

                row_map = {row["id"]: row for row in rec_rows}
                for tid in rec_ids:
                    if tid in row_map:
                        item = row_map[tid]
                        g_list = json.loads(item["genres"] or "[]")
                        genre_note = f"in {' & '.join(g_list[:2])}" if g_list else "in cinema"
                        recs.append({
                            "title_id": item["id"],
                            "tmdb_id": item["tmdb_id"],
                            "media_type": item["media_type"],
                            "title": item["title"],
                            "release_year": item["release_year"],
                            "poster_path": item["poster_path"],
                            "genres": g_list,
                            "overview": item["overview"],
                            "reason": f"Curated for you based on your taste {genre_note}.",
                            "is_on_watchlist": bool(item["is_on_watchlist"]),
                            "vote_average": item["vote_average"],
                            "imdb_rating": item["imdb_rating"] if ("imdb_rating" in item.keys() and item["imdb_rating"]) else item["vote_average"],
                            "imdb_id": item["imdb_id"] if "imdb_id" in item.keys() else None,
                        })

            messages.append({
                "role": r["role"],
                "content": r["content"],
                "recommendations": recs
            })

        return templates.TemplateResponse(request, "chat.html", {
            "request": request,
            "active_tab": "chat",
            "messages": messages,
            "session_id": session_id,
            **stats
        })

    # Default / page refresh: start a brand new chat session
    new_session_id = f"session_{uuid.uuid4().hex[:12]}"
    return templates.TemplateResponse(request, "chat.html", {
        "request": request,
        "active_tab": "chat",
        "messages": [],
        "session_id": new_session_id,
        **stats
    })

@app.get("/deck", response_class=HTMLResponse)
async def page_deck(request: Request, title_id: Optional[int] = None):
    stats = get_global_stats()
    title = None
    if title_id is not None:
        title = await get_deck_title_by_id(title_id)
    if title is None:
        title = await get_next_deck_title()
    return templates.TemplateResponse(request, "deck.html", {
        "request": request,
        "active_tab": "deck",
        "title": title,
        **stats
    })

@app.get("/search", response_class=HTMLResponse)
async def page_search(request: Request, q: Optional[str] = None):
    stats = get_global_stats()
    results = []
    if q and q.strip():
        results = await perform_title_search(q.strip())
    else:
        # Pre-seed top popular titles so search page is lively on load
        with get_db() as conn:
            rows = conn.execute("""
                SELECT t.*, r.score as user_rating, r.aspect_tags as user_aspect_tags,
                       r.notes as user_notes, CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_in_watchlist
                FROM titles t
                LEFT JOIN ratings r ON t.id = r.title_id
                LEFT JOIN watchlist w ON t.id = w.title_id
                ORDER BY t.popularity DESC LIMIT 12
            """).fetchall()
        results = [format_title_row(r) for r in rows]

    return templates.TemplateResponse(request, "search.html", {
        "request": request,
        "active_tab": "search",
        "query": q or "",
        "results": results,
        **stats
    })

@app.get("/library", response_class=HTMLResponse)
async def page_library(request: Request, tab: str = "ratings"):
    stats = get_global_stats()

    ratings_list = []
    watchlist_list = []
    dossier_data = {}

    with get_db() as conn:
        if tab == "ratings" or not tab:
            rows = conn.execute("""
                SELECT r.id, r.score, r.aspect_tags, r.notes, r.created_at,
                       t.id as title_id, t.title, t.media_type, t.release_year,
                       t.poster_path, t.genres, t.director_or_creator
                FROM ratings r
                JOIN titles t ON r.title_id = t.id
                ORDER BY r.created_at DESC
            """).fetchall()
            for r in rows:
                ratings_list.append({
                    "id": r["id"],
                    "title_id": r["title_id"],
                    "score": r["score"],
                    "aspect_tags": json.loads(r["aspect_tags"] or "[]"),
                    "notes": r["notes"],
                    "title": r["title"],
                    "media_type": r["media_type"],
                    "release_year": r["release_year"],
                    "poster_path": r["poster_path"],
                    "genres": json.loads(r["genres"] or "[]"),
                    "director": r["director_or_creator"]
                })

        elif tab == "watchlist":
            rows = conn.execute("""
                SELECT w.id, w.notes, w.created_at,
                       t.id as title_id, t.title, t.media_type, t.release_year,
                       t.poster_path, t.overview, t.genres
                FROM watchlist w
                JOIN titles t ON w.title_id = t.id
                ORDER BY w.created_at DESC
            """).fetchall()
            for w in rows:
                watchlist_list.append({
                    "id": w["id"],
                    "title_id": w["title_id"],
                    "notes": w["notes"],
                    "title": w["title"],
                    "media_type": w["media_type"],
                    "release_year": w["release_year"],
                    "poster_path": w["poster_path"],
                    "overview": w["overview"],
                    "genres": json.loads(w["genres"] or "[]")
                })

        elif tab == "dossier":
            dossier_data = await taste_dossier_service.get_or_update_dossier()

    return templates.TemplateResponse(request, "library.html", {
        "request": request,
        "active_tab": "library",
        "current_tab": tab,
        "ratings": ratings_list,
        "watchlist": watchlist_list,
        "dossier": dossier_data,
        **stats
    })

@app.get("/for-you", response_class=HTMLResponse)
async def page_for_you(request: Request, media_type: str = "movie", refresh: bool = False):
    """Personal shelf: 5 Movies or 5 Series most likely to be liked, from Ratings + Taste Dossier."""
    stats = get_global_stats()
    if media_type not in ("movie", "tv"):
        media_type = "movie"
    result = await orchestrator_service.get_personalized_picks(
        limit=5,
        media_type_preference=media_type,
        force_refresh=refresh,
    )
    return templates.TemplateResponse(request, "for_you.html", {
        "request": request,
        "active_tab": "for-you",
        "picks": result["picks"],
        "message": result["message"],
        "personalized": result.get("personalized", False),
        "media_type": media_type,
        **stats,
        "ratings_count": result.get("ratings_count", stats["ratings_count"]),
    })


# --------------------------------------------------------------------------
# HTMX & API Endpoints
# --------------------------------------------------------------------------

@app.get("/api/for-you", response_class=HTMLResponse)
async def api_for_you(
    request: Request,
    media_type: str = "movie",
    limit: int = 5,
    refresh: bool = False,
    format: Optional[str] = None
):
    """HTMX partial (or JSON with ?format=json) for the personalized 5-Title shelf."""
    if media_type not in ("movie", "tv"):
        media_type = "movie"
    limit = max(1, min(limit, 10))
    result = await orchestrator_service.get_personalized_picks(
        limit=limit,
        media_type_preference=media_type,
        force_refresh=refresh,
    )
    if format == "json" or "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content={
            "picks": result["picks"],
            "message": result["message"],
            "ratings_count": result.get("ratings_count", 0),
            "personalized": result.get("personalized", False),
        })
    return templates.TemplateResponse(request, "components/for_you_results.html", {
        "request": request,
        "picks": result["picks"],
        "message": result["message"],
        "ratings_count": result.get("ratings_count", 0),
        "personalized": result.get("personalized", False),
        "media_type": media_type,
        "is_htmx": True,
    })

@app.post("/api/chat", response_class=HTMLResponse)
async def api_chat(
    request: Request,
    message: str = Form(...),
    session_id: str = Form("default_session"),
    media_type: str = Form("all")
):
    """Processes user chat vibe query via Orchestrator and returns rendered HTML message."""
    result = await orchestrator_service.handle_vibe_query(
        query_text=message,
        session_id=session_id,
        media_type_preference=media_type
    )

    user_html = templates.get_template("components/chat_message.html").render({
        "role": "user",
        "content": message
    })

    assistant_html = templates.get_template("components/chat_message.html").render({
        "role": "assistant",
        "content": result["assistant_message"],
        "recommendations": result["recommendations"]
    })

    return HTMLResponse(content=user_html + assistant_html)

@app.post("/api/chat/clear", response_class=HTMLResponse)
async def api_chat_clear(
    session_id: str = Form("default_session")
):
    """Clears conversation history for the given session."""
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    return HTMLResponse(content="")

@app.get("/api/deck/next", response_class=HTMLResponse)
async def api_deck_next(request: Request):
    title = await get_next_deck_title()
    return templates.TemplateResponse(request, "components/deck_card.html", {
        "request": request,
        "title": title
    })

@app.post("/api/deck/skip/{title_id}", response_class=HTMLResponse)
async def api_deck_skip(request: Request, title_id: int):
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO skipped_titles (title_id) VALUES (?)", (title_id,))
    title = await get_next_deck_title()
    return templates.TemplateResponse(request, "components/deck_card.html", {
        "request": request,
        "title": title
    })

@app.post("/api/deck/watchlist/{title_id}", response_class=HTMLResponse)
async def api_deck_watchlist(request: Request, title_id: int):
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist (title_id) VALUES (?)", (title_id,))
        # Also skip so it advances past the deck
        conn.execute("INSERT OR IGNORE INTO skipped_titles (title_id) VALUES (?)", (title_id,))
    title = await get_next_deck_title()
    return templates.TemplateResponse(request, "components/deck_card.html", {
        "request": request,
        "title": title
    })

class RatingPayload(BaseModel):
    title_id: int
    score: int
    aspect_tags: List[str] = []
    notes: Optional[str] = None

@app.post("/api/ratings")
async def api_save_rating(payload: RatingPayload):
    if not (1 <= payload.score <= 6):
        raise HTTPException(status_code=400, detail="Score must be between 1 and 6")

    with get_db() as conn:
        conn.execute("""
            INSERT INTO ratings (title_id, score, aspect_tags, notes, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(title_id) DO UPDATE SET
                score = excluded.score,
                aspect_tags = excluded.aspect_tags,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
        """, (
            payload.title_id,
            payload.score,
            json.dumps(payload.aspect_tags),
            payload.notes
        ))
        # Remove from watchlist and skipped if previously logged there
        conn.execute("DELETE FROM watchlist WHERE title_id = ?", (payload.title_id,))
        conn.execute("DELETE FROM skipped_titles WHERE title_id = ?", (payload.title_id,))

    # Ensure title has details and embedding for Taste Dossier & recommendations
    asyncio.create_task(_ensure_title_embedding(payload.title_id))

    # Flag taste dossier as needing update
    taste_dossier_service.mark_dirty()
    return JSONResponse(content={"status": "ok", "title_id": payload.title_id, "score": payload.score})

@app.delete("/api/ratings/{rating_id}", response_class=HTMLResponse)
async def api_delete_rating(rating_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM ratings WHERE id = ?", (rating_id,))
    taste_dossier_service.mark_dirty()
    return HTMLResponse(content="")

@app.post("/api/watchlist/toggle/{title_id}", response_class=HTMLResponse)
async def api_toggle_watchlist(title_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM watchlist WHERE title_id = ?", (title_id,)).fetchone()
        if row:
            conn.execute("DELETE FROM watchlist WHERE id = ?", (row["id"],))
            in_watchlist = False
        else:
            conn.execute("INSERT INTO watchlist (title_id) VALUES (?)", (title_id,))
            in_watchlist = True

    fill_class = "fill-notion-primary text-notion-primary" if in_watchlist else ""
    label = "Saved to Watchlist" if in_watchlist else "Save to Watchlist"
    return HTMLResponse(f"""
        <button type="button"
                hx-post="/api/watchlist/toggle/{title_id}"
                hx-swap="outerHTML"
                aria-label="{label}" title="{label}"
                class="p-2 rounded-lg border border-notion-hairline text-notion-slate hover:bg-notion-surface">
            <i data-lucide="bookmark" style="width:15px;height:15px;" class="{fill_class}"></i>
        </button>
    """)

@app.delete("/api/watchlist/{watchlist_id}", response_class=HTMLResponse)
async def api_delete_watchlist(watchlist_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM watchlist WHERE id = ?", (watchlist_id,))
    return HTMLResponse(content="")

@app.get("/api/search", response_class=HTMLResponse)
async def api_search(request: Request, q: str = Query("", alias="q")):
    results = await perform_title_search(q.strip())
    return templates.TemplateResponse(request, "components/search_results.html", {
        "request": request,
        "query": q,
        "results": results
    })

@app.get("/api/stats")
async def api_stats():
    stats = get_global_stats()
    return HTMLResponse(f"""
        <span class="flex items-center gap-1.5"><i data-lucide="star" style="width:13px;height:13px;" class="text-notion-primary"></i><strong id="nav-ratings-count" class="text-notion-ink font-medium">{stats['ratings_count']}</strong> rated</span>
        <span class="text-notion-hairlineStrong" aria-hidden="true">·</span>
        <span class="flex items-center gap-1.5"><i data-lucide="bookmark" style="width:13px;height:13px;" class="text-notion-purple"></i><strong id="nav-watchlist-count" class="text-notion-ink font-medium">{stats['watchlist_count']}</strong> saved</span>
    """)

@app.post("/api/dossier/regenerate", response_class=HTMLResponse)
async def api_dossier_regenerate(request: Request):
    dossier = await taste_dossier_service.get_or_update_dossier(force=True)
    return HTMLResponse(f"""
        <div class="cm-card p-5 text-[15px] leading-relaxed text-notion-charcoal">
            {escape(dossier['full_summary'] or 'Rate at least 3 Titles and your Dossier will appear here.')}
        </div>
        <div class="grid sm:grid-cols-2 gap-3 mt-3">
            <div class="rounded-xl bg-notion-mint p-5">
                <p class="cm-eyebrow" style="color:#1aae39;">Loves</p>
                <div class="flex flex-wrap gap-1.5 mt-2">
                    {''.join(f'<span class="px-2 py-1 rounded-md bg-notion-canvas border border-notion-hairline text-[13px] font-medium">{escape(item)}</span>' for item in dossier['core_loves'])}
                </div>
            </div>
            <div class="rounded-xl bg-notion-rose p-5">
                <p class="cm-eyebrow" style="color:#a02e6d;">Deal-breakers</p>
                <div class="flex flex-wrap gap-1.5 mt-2">
                    {''.join(f'<span class="px-2 py-1 rounded-md bg-notion-canvas border border-notion-hairline text-[13px] font-medium">{escape(item)}</span>' for item in dossier['deal_breakers'])}
                </div>
            </div>
            <div class="rounded-xl bg-notion-sky p-5">
                <p class="cm-eyebrow" style="color:#0075de;">Atmosphere</p>
                <div class="flex flex-wrap gap-1.5 mt-2">
                    {''.join(f'<span class="px-2 py-1 rounded-md bg-notion-canvas border border-notion-hairline text-[13px] font-medium">{escape(item)}</span>' for item in dossier['atmospheric_preferences'])}
                </div>
            </div>
            <div class="rounded-xl bg-notion-lavender p-5">
                <p class="cm-eyebrow" style="color:#391c57;">Creators</p>
                <div class="flex flex-wrap gap-1.5 mt-2">
                    {''.join(f'<span class="px-2 py-1 rounded-md bg-notion-canvas border border-notion-hairline text-[13px] font-medium">{escape(item)}</span>' for item in dossier['creator_affinities'])}
                </div>
            </div>
        </div>
    """)


# --------------------------------------------------------------------------
# Internal Data Helpers
# --------------------------------------------------------------------------

async def get_deck_title_by_id(title_id: int) -> Optional[dict]:
    """Fetches a specific Title for the Rating Deck (e.g. selected from Search)."""
    with get_db() as conn:
        row = conn.execute("SELECT t.* FROM titles t WHERE t.id = ?", (title_id,)).fetchone()

    if not row:
        return None

    title_dict = format_title_row(row)

    # If director or cast is missing and TMDB API key is active, fetch details on-demand
    if settings.TMDB_API_KEY and (not title_dict.get("director_or_creator") or not title_dict.get("cast_top")):
        try:
            details = await tmdb_service.get_title_details(title_dict["tmdb_id"], title_dict["media_type"])
            if details:
                title_dict["director_or_creator"] = details.get("director_or_creator")
                title_dict["cast_top"] = details.get("cast_top") or []
                title_dict["imdb_id"] = details.get("imdb_id")
                title_dict["imdb_rating"] = details.get("imdb_rating") or details.get("vote_average", 0.0)
                with get_db() as conn:
                    conn.execute("""
                        UPDATE titles SET
                            director_or_creator = ?,
                            cast_top = ?,
                            imdb_id = ?,
                            imdb_rating = ?
                        WHERE id = ?
                    """, (
                        details.get("director_or_creator"),
                        json.dumps(details.get("cast_top") or []),
                        details.get("imdb_id"),
                        details.get("imdb_rating") or details.get("vote_average", 0.0),
                        title_id
                    ))
        except Exception as e:
            logger.error(f"Error fetching on-demand details for title #{title_id}: {e}")

    return title_dict

DECK_REFILL_LOW_WATERMARK = 15  # Proactively refill when remaining unseen titles drop below 15
DECK_BUFFER_TARGET = 30
DECK_SUGGESTION_CADENCE = 3     # Occasional cadence: every 3rd card can be a personalized suggestion
DECK_REFILL_PAGES_PER_ENDPOINT = 2
DECK_REFILL_MAX_WINDOWS = 5
DECK_MAX_TMDB_PAGE = 500
DECK_REFILL_INSERT_LIMIT = 16

_deck_refill_lock = asyncio.Lock()


def _get_deck_cursor() -> int:
    """Returns the next TMDB page the Deck refill should start from."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deck_refill_state (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                next_page INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("INSERT OR IGNORE INTO deck_refill_state (id, next_page) VALUES (1, 1)")
        row = conn.execute("SELECT next_page FROM deck_refill_state WHERE id = 1").fetchone()
        return int(row["next_page"] or 1) if row else 1


def _advance_deck_cursor(pages_consumed: int) -> int:
    """Advances the Deck refill cursor, wrapping past the TMDB page cap."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deck_refill_state (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                next_page INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("INSERT OR IGNORE INTO deck_refill_state (id, next_page) VALUES (1, 1)")
        row = conn.execute("SELECT next_page FROM deck_refill_state WHERE id = 1").fetchone()
        current = int(row["next_page"] or 1) if row else 1
        nxt = current + pages_consumed
        if nxt > DECK_MAX_TMDB_PAGE:
            nxt = 1
        conn.execute("UPDATE deck_refill_state SET next_page = ? WHERE id = 1", (nxt,))
        return nxt


def _get_and_increment_deck_serve_count() -> int:
    """Returns current deck serve count and increments it in SQLite for persistent cadence."""
    with get_db() as conn:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(deck_refill_state)").fetchall()]
        if "serve_count" not in cols:
            conn.execute("ALTER TABLE deck_refill_state ADD COLUMN serve_count INTEGER NOT NULL DEFAULT 0")
        row = conn.execute("SELECT serve_count FROM deck_refill_state WHERE id = 1").fetchone()
        count = int(row["serve_count"] or 0) if (row and "serve_count" in row.keys() and row["serve_count"] is not None) else 0
        conn.execute("UPDATE deck_refill_state SET serve_count = ? WHERE id = 1", (count + 1,))
        return count


async def _cache_items_concurrently(items: list, existing_keys: set, limit: int = DECK_REFILL_INSERT_LIMIT) -> int:
    """Persists fresh TMDB list items with details + embeddings concurrently, up to `limit`."""
    fresh = [i for i in items if (i["tmdb_id"], i["media_type"]) not in existing_keys][:limit]
    if not fresh:
        return 0

    sem = asyncio.Semaphore(4)

    async def _process_item(item):
        async with sem:
            try:
                details = await tmdb_service.get_title_details(item["tmdb_id"], item["media_type"])
                full = details or item
                if not details:
                    full = {**item, "director_or_creator": None, "cast_top": []}
                embed_text = embedding_service.build_title_embedding_text(full)
                vec = await embedding_service.get_embedding(embed_text)
                blob = embedding_service.vec_to_bytes(vec)
                return (full, blob, len(vec))
            except Exception as e:
                logger.error(f"Error processing Title {item.get('title')}: {e}")
                return None

    results = await asyncio.gather(*(_process_item(it) for it in fresh))
    inserted = 0
    with get_db() as conn:
        for res in results:
            if not res:
                continue
            full, blob, dim = res
            try:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO titles (
                        tmdb_id, media_type, title, original_title, release_year,
                        overview, poster_path, backdrop_path, genres,
                        director_or_creator, cast_top, vote_average, vote_count,
                        popularity, imdb_id, imdb_rating, embedding, embedding_dim, embedding_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    full["tmdb_id"], full["media_type"], full["title"],
                    full.get("original_title"), full.get("release_year"),
                    full.get("overview"), full.get("poster_path"),
                    full.get("backdrop_path"), json.dumps(full.get("genres") or []),
                    full.get("director_or_creator"), json.dumps(full.get("cast_top") or []),
                    full.get("vote_average", 0.0), full.get("vote_count", 0),
                    full.get("popularity", 0.0), full.get("imdb_id"),
                    full.get("imdb_rating") or full.get("vote_average", 0.0),
                    blob, dim, embedding_service.model_key
                ))
                if cursor.lastrowid:
                    inserted += 1
                    existing_keys.add((full["tmdb_id"], full["media_type"]))
            except Exception as e:
                logger.error(f"Error inserting Title {full.get('title')}: {e}")
    return inserted


async def refill_deck_with_preferences_and_famous() -> int:
    """Refills the local catalog with preference-targeted titles and famous titles.

    Combines:
    1. TMDB recommendations based on user's highest rated titles (score >= 4).
    2. Genre discovery from user's most frequently loved genres.
    3. Broad famous / top-rated titles to keep catalog diverse.
    """
    if not settings.TMDB_API_KEY:
        return 0

    with get_db() as conn:
        existing_keys = {
            (r["tmdb_id"], r["media_type"])
            for r in conn.execute("SELECT tmdb_id, media_type FROM titles").fetchall()
        }
        liked_rows = conn.execute("""
            SELECT r.score, t.tmdb_id, t.media_type, t.genres, t.director_or_creator
            FROM ratings r
            JOIN titles t ON r.title_id = t.id
            WHERE r.score >= 4
            ORDER BY r.score DESC, r.created_at DESC
        """).fetchall()

    candidates: List[dict] = []

    # --- 1. Preference-Based Candidates ---
    if liked_rows:
        from collections import Counter
        from app.services.tmdb import GENRE_NAME_TO_ID

        top_loved = [r for r in liked_rows if r["score"] >= 5][:3] or liked_rows[:3]
        for item in top_loved:
            try:
                recs = await tmdb_service.get_title_recommendations(
                    tmdb_id=item["tmdb_id"],
                    media_type=item["media_type"],
                    limit=8
                )
                for r in recs:
                    if (r["tmdb_id"], r["media_type"]) not in existing_keys:
                        candidates.append(r)
            except Exception as e:
                logger.error(f"Error getting recommendations for {item['tmdb_id']}: {e}")

        genre_counter = Counter()
        for r in liked_rows:
            try:
                for g in json.loads(r["genres"] or "[]"):
                    genre_counter[g] += 1
            except Exception:
                pass

        for g_name, _ in genre_counter.most_common(2):
            gid = GENRE_NAME_TO_ID.get(g_name.lower())
            if gid:
                for m_type in ("movie", "tv"):
                    try:
                        disc = await tmdb_service.discover_titles(
                            media_type=m_type,
                            with_genres=str(gid),
                            sort_by="popularity.desc",
                            vote_count_gte=250,
                            vote_average_gte=6.5,
                            limit=8
                        )
                        for d in disc:
                            if (d["tmdb_id"], d["media_type"]) not in existing_keys:
                                candidates.append(d)
                    except Exception as e:
                        logger.error(f"Error discovering {g_name} {m_type}: {e}")

    # --- 2. Famous / Breadth Candidates ---
    start_page = _get_deck_cursor()
    try:
        famous = await tmdb_service.get_famous_titles(
            min_vote_count=DECK_MIN_VOTE_COUNT,
            min_vote_average=DECK_MIN_VOTE_AVERAGE,
            pages_per_endpoint=DECK_REFILL_PAGES_PER_ENDPOINT,
            start_page=start_page,
            limit=20
        )
        for f in famous:
            if (f["tmdb_id"], f["media_type"]) not in existing_keys:
                candidates.append(f)
    except Exception as e:
        logger.error(f"Error fetching famous titles for refill: {e}")
    finally:
        _advance_deck_cursor(DECK_REFILL_PAGES_PER_ENDPOINT)

    if not candidates:
        logger.warning("Deck refill: no fresh candidates found.")
        return 0

    # Deduplicate within batch
    deduped = []
    seen_in_batch = set()
    for c in candidates:
        k = (c["tmdb_id"], c["media_type"])
        if k not in seen_in_batch and k not in existing_keys:
            seen_in_batch.add(k)
            deduped.append(c)

    inserted = await _cache_items_concurrently(deduped, existing_keys, limit=DECK_REFILL_INSERT_LIMIT)
    logger.info(f"Deck refill: cached {inserted} fresh titles (preferences + famous).")
    return inserted


async def refill_deck_with_famous_titles() -> int:
    """Wrapper preserved for backward compatibility; calls preference+famous refill."""
    return await refill_deck_with_preferences_and_famous()


def schedule_deck_refill_if_needed():
    """Triggers background refill without blocking the current request if not already running."""
    async def _runner():
        if _deck_refill_lock.locked():
            return
        async with _deck_refill_lock:
            try:
                await refill_deck_with_preferences_and_famous()
            except Exception as e:
                logger.error(f"Background deck refill error: {e}")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_runner())
    except RuntimeError:
        pass


async def get_next_deck_title() -> Optional[dict]:
    """Pulls the next Title for the Deck — never completes.

    Endless & Personalized behavior:
    1. Proactively triggers background refill if remaining unseen titles < DECK_REFILL_LOW_WATERMARK.
    2. On cadence (every 3rd card), serves a personalized suggestion based on the user's
       ratings / taste vector if one is available.
    3. Otherwise serves widely recognized unrated titles.
    4. Refills when the unseen quality pool is exhausted.
    5. Finally recycles the oldest skipped quality Title.
    """
    with get_db() as conn:
        remaining = conn.execute("""
            SELECT COUNT(*)
            FROM titles t
            LEFT JOIN ratings r ON t.id = r.title_id
            LEFT JOIN skipped_titles s ON t.id = s.title_id
            LEFT JOIN watchlist w ON t.id = w.title_id
            WHERE r.id IS NULL AND s.id IS NULL AND w.id IS NULL
              AND t.vote_count >= ? AND t.vote_average >= ?
        """, (DECK_MIN_VOTE_COUNT, DECK_MIN_VOTE_AVERAGE)).fetchone()[0]

    # Proactive non-blocking refill before running out!
    if remaining < DECK_REFILL_LOW_WATERMARK:
        schedule_deck_refill_if_needed()

    serve_count = _get_and_increment_deck_serve_count()

    # Step 1: Occasional personalized suggestion based on preferences
    if serve_count % DECK_SUGGESTION_CADENCE == 0:
        try:
            suggestion = await orchestrator_service.get_deck_suggestion()
            if suggestion:
                return suggestion
        except Exception as e:
            logger.error(f"Error fetching personalized deck suggestion: {e}")

    # Step 2: High quality / famous unrated titles
    with get_db() as conn:
        row = conn.execute("""
            SELECT t.*
            FROM titles t
            LEFT JOIN ratings r ON t.id = r.title_id
            LEFT JOIN skipped_titles s ON t.id = s.title_id
            LEFT JOIN watchlist w ON t.id = w.title_id
            WHERE r.id IS NULL AND s.id IS NULL AND w.id IS NULL
              AND t.vote_count >= ? AND t.vote_average >= ?
            ORDER BY t.popularity DESC, RANDOM()
            LIMIT 1
        """, (DECK_MIN_VOTE_COUNT, DECK_MIN_VOTE_AVERAGE)).fetchone()

    if row:
        return format_title_row(row)

    # Step 3: Refill the quality pool synchronously once it is exhausted.
    if settings.TMDB_API_KEY:
        await refill_deck_with_preferences_and_famous()
        with get_db() as conn:
            row = conn.execute("""
                SELECT t.*
                FROM titles t
                LEFT JOIN ratings r ON t.id = r.title_id
                LEFT JOIN skipped_titles s ON t.id = s.title_id
                LEFT JOIN watchlist w ON t.id = w.title_id
                WHERE r.id IS NULL AND s.id IS NULL AND w.id IS NULL
                  AND t.vote_count >= ? AND t.vote_average >= ?
                ORDER BY t.popularity DESC, RANDOM()
                LIMIT 1
            """, (DECK_MIN_VOTE_COUNT, DECK_MIN_VOTE_AVERAGE)).fetchone()
        if row:
            return format_title_row(row)

    # Step 4: Recycle the oldest skipped quality Title so low-signal Titles are
    # never served merely because the fresh pool ran out.
    with get_db() as conn:
        row = conn.execute("""
            SELECT t.*
            FROM titles t
            JOIN skipped_titles s ON t.id = s.title_id
            LEFT JOIN ratings r ON t.id = r.title_id
            LEFT JOIN watchlist w ON t.id = w.title_id
            WHERE r.id IS NULL AND w.id IS NULL
              AND t.vote_count >= ? AND t.vote_average >= ?
            ORDER BY s.created_at ASC
            LIMIT 1
        """, (DECK_MIN_VOTE_COUNT, DECK_MIN_VOTE_AVERAGE)).fetchone()
        if row:
            conn.execute("DELETE FROM skipped_titles WHERE title_id = ?", (row["id"],))

    if not row:
        return None

    return format_title_row(row)

async def _ensure_title_embedding(title_id: int):
    """Ensures a rated Title has full details and vector embedding in SQLite."""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM titles WHERE id = ?", (title_id,)).fetchone()
        if not row:
            return
        if row["embedding"] and row["director_or_creator"] and row["embedding_model"] == embedding_service.model_key:
            return

        title_dict = format_title_row(row)
        details = None
        if settings.TMDB_API_KEY and (not title_dict.get("director_or_creator") or not title_dict.get("cast_top")):
            details = await tmdb_service.get_title_details(title_dict["tmdb_id"], title_dict["media_type"])

        full = {**title_dict, **(details or {})}
        embed_text = embedding_service.build_title_embedding_text(full)
        vec = await embedding_service.get_embedding(embed_text)
        blob = embedding_service.vec_to_bytes(vec)

        with get_db() as conn:
            conn.execute("""
                UPDATE titles SET
                    overview = COALESCE(?, overview),
                    genres = COALESCE(?, genres),
                    director_or_creator = COALESCE(?, director_or_creator),
                    cast_top = COALESCE(?, cast_top),
                    imdb_id = COALESCE(?, imdb_id),
                    imdb_rating = COALESCE(?, imdb_rating),
                    embedding = ?,
                    embedding_dim = ?,
                    embedding_model = ?
                WHERE id = ?
            """, (
                (details.get("overview") if details else None),
                (json.dumps(details["genres"]) if details and "genres" in details else None),
                (details.get("director_or_creator") if details else None),
                (json.dumps(details.get("cast_top") or []) if details else None),
                (details.get("imdb_id") if details else None),
                (details.get("imdb_rating") if details else None),
                blob,
                len(vec),
                embedding_service.model_key,
                title_id
            ))
            conn.execute("DELETE FROM for_you_cache")
            conn.execute("UPDATE taste_dossiers SET is_dirty = 1")
            conn.execute("UPDATE ratings_state SET revision = revision + 1 WHERE id = 1")
    except Exception as e:
        logger.error(f"Error ensuring embedding for title #{title_id}: {e}")

async def perform_title_search(query_text: str) -> List[dict]:
    """Searches local database and falls back to live TMDB search."""
    if not query_text:
        return []

    results = []
    # 1. Search local SQLite
    with get_db() as conn:
        rows = conn.execute("""
            SELECT t.*, r.score as user_rating, r.aspect_tags as user_aspect_tags,
                   r.notes as user_notes, CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_in_watchlist
            FROM titles t
            LEFT JOIN ratings r ON t.id = r.title_id
            LEFT JOIN watchlist w ON t.id = w.title_id
            WHERE t.title LIKE ? OR t.original_title LIKE ?
            ORDER BY t.popularity DESC
            LIMIT 12
        """, (f"%{query_text}%", f"%{query_text}%")).fetchall()

    for r in rows:
        results.append(format_title_row(r))

    # 2. If fewer than 4 local results and TMDB API key is active, search TMDB
    if len(results) < 4 and settings.TMDB_API_KEY:
        tmdb_items = await tmdb_service.search_titles(query_text)
        existing_keys = {
            (res.get("tmdb_id"), res.get("media_type"))
            for res in results
            if res.get("tmdb_id") and res.get("media_type")
        }

        needed = 12 - len(results)
        fresh_items = [
            item for item in tmdb_items
            if (item["tmdb_id"], item["media_type"]) not in existing_keys
        ][:needed]

        if fresh_items:
            with get_db() as conn:
                for item in fresh_items:
                    existing = conn.execute("""
                        SELECT t.*, r.score as user_rating, r.aspect_tags as user_aspect_tags,
                               r.notes as user_notes, CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_in_watchlist
                        FROM titles t
                        LEFT JOIN ratings r ON t.id = r.title_id
                        LEFT JOIN watchlist w ON t.id = w.title_id
                        WHERE t.tmdb_id = ? AND t.media_type = ?
                    """, (item["tmdb_id"], item["media_type"])).fetchone()

                    if existing:
                        results.append(format_title_row(existing))
                    else:
                        cursor = conn.execute("""
                            INSERT OR IGNORE INTO titles (
                                tmdb_id, media_type, title, original_title, release_year,
                                overview, poster_path, backdrop_path, genres,
                                vote_average, vote_count, popularity
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            item["tmdb_id"], item["media_type"], item["title"],
                            item.get("original_title"), item.get("release_year"),
                            item.get("overview"), item.get("poster_path"),
                            item.get("backdrop_path"), json.dumps(item.get("genres") or []),
                            item.get("vote_average", 0.0), item.get("vote_count", 0),
                            item.get("popularity", 0.0)
                        ))
                        new_id = cursor.lastrowid
                        if not new_id:
                            row_check = conn.execute(
                                "SELECT id FROM titles WHERE tmdb_id = ? AND media_type = ?",
                                (item["tmdb_id"], item["media_type"])
                            ).fetchone()
                            new_id = row_check["id"] if row_check else 0

                        results.append({
                            **item,
                            "id": new_id,
                            "user_rating": None,
                            "user_aspect_tags": [],
                            "user_notes": None,
                            "is_in_watchlist": False,
                        })

    return results

def format_title_row(row) -> dict:
    """Formats a SQLite row or title dict into a standardized title dict."""
    if isinstance(row, dict) and "genres" in row and isinstance(row["genres"], list):
        # Already formatted dictionary
        return row

    is_dict = isinstance(row, dict)
    keys = row.keys() if hasattr(row, "keys") else ()

    raw_genres = row["genres"] if (is_dict or "genres" in keys) else "[]"
    genres = json.loads(raw_genres) if isinstance(raw_genres, str) else (raw_genres or [])

    raw_cast = row["cast_top"] if (is_dict or "cast_top" in keys) else "[]"
    cast_top = json.loads(raw_cast) if isinstance(raw_cast, str) else (raw_cast or [])

    raw_tags = row["user_aspect_tags"] if (is_dict or "user_aspect_tags" in keys) else "[]"
    user_aspect_tags = json.loads(raw_tags) if isinstance(raw_tags, str) else (raw_tags or [])

    return {
        "id": row["id"],
        "tmdb_id": row["tmdb_id"],
        "media_type": row["media_type"],
        "title": row["title"],
        "original_title": row["original_title"] if (is_dict or "original_title" in keys) else None,
        "release_year": row["release_year"],
        "overview": row["overview"],
        "poster_path": row["poster_path"],
        "backdrop_path": row["backdrop_path"] if (is_dict or "backdrop_path" in keys) else None,
        "genres": genres,
        "director_or_creator": row["director_or_creator"] if (is_dict or "director_or_creator" in keys) else None,
        "cast_top": cast_top,
        "vote_average": row["vote_average"],
        "vote_count": row["vote_count"],
        "popularity": row["popularity"],
        "user_rating": row["user_rating"] if (is_dict or "user_rating" in keys) else None,
        "user_aspect_tags": user_aspect_tags,
        "user_notes": row["user_notes"] if (is_dict or "user_notes" in keys) else None,
        "is_in_watchlist": bool(row["is_in_watchlist"]) if (is_dict or "is_in_watchlist" in keys) else False,
        "is_suggestion": bool(row["is_suggestion"]) if (is_dict or "is_suggestion" in keys) else False,
        "suggestion_reason": row["suggestion_reason"] if (is_dict or "suggestion_reason" in keys) else None,
        "match_score": row["match_score"] if (is_dict or "match_score" in keys) else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.APP_PORT, reload=settings.DEBUG)
