"""Opt-in live evaluation using public catalog metadata and synthetic ratings.

Run: python -m tests.evaluate_ai_pipeline --live
Uses a fresh temporary database. Never copies personal ratings, notes or chats.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import sqlite3
import tempfile
import time

from app.config import settings
from app.database import get_db, init_db
from app.services.embeddings import embedding_service
from app.services.orchestrator import OrchestratorService
from app.services.taste_dossier import TasteDossierService
from app.services.tmdb import tmdb_service


async def evaluate():
    source = settings.DATABASE_PATH.resolve()
    # Only the public catalog is read; source is explicitly read-only.
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        catalog = [dict(r) for r in conn.execute("SELECT * FROM titles")]
    root = Path(".test-tmp").resolve()
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="live-taste-", dir=root))
    settings.DATABASE_PATH = output / "evaluation.db"
    init_db()
    with get_db() as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(titles)")]
        conn.executemany(
            f"INSERT INTO titles ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            [[r.get(c) for c in columns] for r in catalog],
        )
        for name, score, tags, note in [
            ("La La Land", 6, ["Soundtrack", "Emotion"], "Love the music and bittersweet romance; sad endings are fine."),
            ("Before Sunrise", 6, ["Dialogue"], "I love intimate, talkative romance, even when it moves slowly."),
            ("Before Sunset", 5, ["Dialogue"], "Thoughtful romantic conversation is what I liked."),
            ("The Office", 6, ["Humor", "Characters"], "For series I prefer warm workplace ensemble comedy."),
            ("Ted Lasso", 5, ["Characters"], "The kindness and ensemble friendships work for me."),
            ("The Conjuring", 1, ["Atmosphere"], "I dislike supernatural horror and jump scares."),
            ("It", 1, ["Atmosphere"], "Horror and gore are not what I want to watch."),
        ]:
            row = conn.execute("SELECT id FROM titles WHERE title=?", (name,)).fetchone()
            assert row, f"Missing evaluation seed: {name}"
            conn.execute("INSERT INTO ratings (title_id,score,aspect_tags,notes) VALUES (?,?,?,?)",
                         (row[0], score, json.dumps(tags), note))
    report = {"model": settings.OPENROUTER_MODEL, "catalog_size": len(catalog), "synthetic_ratings": 7,
              "checks": {}, "outputs": {}}

    async def stage(name, call):
        start = time.monotonic()
        print(f"Running {name}...", flush=True)
        value = await call
        report["outputs"][name] = value
        print(f"Finished {name} in {time.monotonic()-start:.1f}s", flush=True)
        return value

    vector = await embedding_service.get_embedding("warm bittersweet romance and thoughtful conversation")
    report["checks"]["cloud_embedding_valid"] = len(vector) == settings.EMBEDDING_DIM
    dossier = await stage("dossier", TasteDossierService().get_or_update_dossier())
    report["checks"]["dossier_synthesized"] = bool(dossier["updated_at"] or (not dossier["is_dirty"] and dossier["core_loves"]))
    svc = OrchestratorService()
    model_selections = []
    create = svc._get_client().chat.completions.create

    async def record_selection(**kwargs):
        response = await create(**kwargs)
        data = json.loads(response.choices[0].message.content)
        if "picks" in data or "recommendations" in data:
            model_selections.extend(data.get("picks", data.get("recommendations", [])))
        return response

    svc._client.chat.completions.create = record_selection
    movies = await stage("movies", svc.get_personalized_picks(media_type_preference="movie", limit=5))
    series = await stage("series", svc.get_personalized_picks(media_type_preference="tv", limit=5))
    chat = await stage("chat", svc.handle_vibe_query("A thoughtful romantic movie with good dialogue. No horror.", "romance", "movie"))
    followup = await stage("followup", svc.handle_vibe_query("More like the first pick, but from before 2000. Keep the same exclusions.", "romance", "movie"))
    fresh = await stage("fresh_session_intent", svc._parse_query_intent("A science fiction series", "tv", []))
    details = await stage("tmdb_details", tmdb_service.get_title_details(313369, "movie"))
    with get_db() as conn:
        conn.execute("UPDATE titles SET embedding=NULL, embedding_model=NULL WHERE tmdb_id=313369 AND media_type='movie'")
    discovered = await stage("tmdb_discovery", svc._discover_and_cache_tmdb({
        "person": "Damien Chazelle", "media_type": "movie", "genres": ["Romance"]}))
    with get_db() as conn:
        rated = {r[0] for r in conn.execute("SELECT title_id FROM ratings")}
        saved = conn.execute("SELECT COUNT(*) FROM chat_messages WHERE session_id='romance'").fetchone()[0]
        title_names = {r[0]: r[1] for r in conn.execute("SELECT id,title FROM titles")}
        repaired = conn.execute("SELECT id,embedding_model FROM titles WHERE tmdb_id=313369 AND media_type='movie'").fetchone()
    for name, picks, media in [("movies", movies["picks"], "movie"), ("series", series["picks"], "tv"),
                                ("chat", chat["recommendations"], "movie"), ("followup", followup["recommendations"], "movie")]:
        report["checks"][name+"_valid_unique_unseen"] = bool(picks) and len({p["title_id"] for p in picks}) == len(picks) and all(p["title_id"] not in rated and p["media_type"] == media for p in picks)
        report["checks"][name+"_avoids_horror"] = all("Horror" not in p["genres"] for p in picks)
    report["checks"]["followup_year"] = all(p["release_year"] and p["release_year"] < 2000 for p in followup["recommendations"])
    report["checks"]["session_persisted"] = saved == 4
    report["checks"]["fresh_session_no_romance_constraint"] = fresh["media_type"] == "tv" and "Romance" not in fresh["genres"]
    report["checks"]["tmdb_resolves_title"] = bool(details and details["title"] == "La La Land")
    report["checks"]["discovery_repairs_embedding"] = repaired[0] in discovered and repaired[1] == embedding_service.model_key
    invalid = [p for p in model_selections if p.get("title") != title_names.get(p.get("title_id"))]
    report["diagnostics"] = {"raw_model_identity_errors": len(invalid), "invalid_model_selections": invalid}
    served = [p for value in (movies, series, chat, followup)
              for p in value.get("picks", value.get("recommendations", []))]
    report["checks"]["invalid_model_selections_not_served"] = all(
        p.get("reason") not in {bad.get("reason") for bad in invalid} for p in served)
    report["outputs"]["model_selections"] = model_selections
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "report": str(report_path)}, indent=2), flush=True)
    return all(report["checks"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make billable API calls using configured credentials")
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to opt into API calls.")
    if not settings.OPENROUTER_API_KEY or not settings.TMDB_API_KEY:
        parser.error("Both OpenRouter and TMDB credentials must be configured.")
    # Provider exceptions can include URLs with credentials; keep them out of artifacts.
    logging.disable(logging.CRITICAL)
    try:
        passed = asyncio.run(evaluate())
    except Exception as error:
        print(f"Evaluation could not complete: {type(error).__name__}", flush=True)
        raise SystemExit(1) from None
    raise SystemExit(0 if passed else 1)
