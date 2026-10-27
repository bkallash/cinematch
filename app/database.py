import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Iterable
from contextlib import contextmanager
from app.config import settings

logger = logging.getLogger(__name__)

TITLE_COLUMNS = (
    "tmdb_id", "media_type", "title", "original_title", "release_year", "overview",
    "poster_path", "backdrop_path", "genres", "director_or_creator", "cast_top",
    "vote_average", "vote_count", "popularity", "imdb_id", "imdb_rating",
    "embedding", "embedding_dim", "embedding_model",
)

def parse_utc_timestamp(ts: Any) -> Optional[datetime]:
    """Safely parse SQLite or ISO timestamp string to timezone-aware UTC datetime."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DATABASE_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

@contextmanager
def get_db():
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_title(conn: sqlite3.Connection, item: Dict[str, Any], embedding: Optional[tuple] = None,
                 update: Iterable[str] = ()) -> int:
    """Insert a TMDB-shaped title dict and return its row id.

    `embedding` is (blob, dim, model_key). On an existing (tmdb_id, media_type),
    only the columns named in `update` are overwritten.
    """
    blob, dim, model = embedding or (None, 0, None)
    values = (
        item["tmdb_id"], item["media_type"], item["title"], item.get("original_title"),
        item.get("release_year"), item.get("overview"), item.get("poster_path"),
        item.get("backdrop_path"), json.dumps(item.get("genres") or []),
        item.get("director_or_creator"), json.dumps(item.get("cast_top") or []),
        item.get("vote_average", 0.0), item.get("vote_count", 0), item.get("popularity", 0.0),
        item.get("imdb_id"), item.get("imdb_rating") or item.get("vote_average", 0.0),
        blob, dim, model,
    )
    update = list(update)
    conflict = (f"DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in update)}" if update else "DO NOTHING")
    conn.execute(f"""
        INSERT INTO titles ({", ".join(TITLE_COLUMNS)}) VALUES ({", ".join("?" * len(TITLE_COLUMNS))})
        ON CONFLICT(tmdb_id, media_type) {conflict}
    """, values)
    return conn.execute("SELECT id FROM titles WHERE tmdb_id = ? AND media_type = ?",
                        (item["tmdb_id"], item["media_type"])).fetchone()[0]


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> set:
    """Add columns that older databases lack; return the names that were added."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    added = set()
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            added.add(name)
    return added


def init_db():
    """Create tables and indexes if they do not exist."""
    settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS titles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tmdb_id INTEGER NOT NULL,
            media_type TEXT NOT NULL CHECK(media_type IN ('movie', 'tv')),
            title TEXT NOT NULL,
            original_title TEXT,
            release_year INTEGER,
            overview TEXT,
            poster_path TEXT,
            backdrop_path TEXT,
            genres TEXT DEFAULT '[]',               -- JSON list of strings
            director_or_creator TEXT,
            cast_top TEXT DEFAULT '[]',             -- JSON list of top actors
            vote_average REAL DEFAULT 0.0,
            vote_count INTEGER DEFAULT 0,
            popularity REAL DEFAULT 0.0,
            imdb_id TEXT,
            imdb_rating REAL,
            embedding BLOB,                         -- float32 numpy array bytes
            embedding_dim INTEGER DEFAULT 0,
            embedding_model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tmdb_id, media_type)
        );

        CREATE INDEX IF NOT EXISTS idx_titles_popularity ON titles(popularity DESC);
        CREATE INDEX IF NOT EXISTS idx_titles_vote_count ON titles(vote_count DESC);
        CREATE INDEX IF NOT EXISTS idx_titles_title ON titles(title COLLATE NOCASE);

        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title_id INTEGER NOT NULL REFERENCES titles(id) ON DELETE CASCADE,
            score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 6),
            aspect_tags TEXT DEFAULT '[]',          -- JSON list of strings
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title_id)
        );

        CREATE INDEX IF NOT EXISTS idx_ratings_score ON ratings(score);

        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title_id INTEGER NOT NULL REFERENCES titles(id) ON DELETE CASCADE,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title_id)
        );

        CREATE TABLE IF NOT EXISTS skipped_titles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title_id INTEGER NOT NULL REFERENCES titles(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title_id)
        );

        CREATE TABLE IF NOT EXISTS for_you_skips (
            user_id TEXT NOT NULL DEFAULT 'default_user',
            title_id INTEGER NOT NULL REFERENCES titles(id) ON DELETE CASCADE,
            PRIMARY KEY (user_id, title_id)
        );

        CREATE TABLE IF NOT EXISTS taste_dossiers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default_user',
            core_loves TEXT DEFAULT '[]',            -- JSON list of themes/motifs
            deal_breakers TEXT DEFAULT '[]',         -- JSON list of dislikes
            creator_affinities TEXT DEFAULT '[]',    -- JSON list of directors/actors
            atmospheric_preferences TEXT DEFAULT '[]',
            narrative_tropes TEXT DEFAULT '[]',
            full_summary TEXT DEFAULT '',
            ratings_count_at_synthesis INTEGER DEFAULT 0,
            is_dirty INTEGER DEFAULT 1,
            evidence_version INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id)
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL DEFAULT 'default_session',
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            recommended_title_ids TEXT DEFAULT '[]', -- JSON list of title_ids
            recommendation_reasons TEXT DEFAULT '{}', -- JSON {title_id: reason}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, created_at);

        CREATE TABLE IF NOT EXISTS deck_refill_state (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            next_page INTEGER NOT NULL DEFAULT 1,
            serve_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS for_you_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default_user',
            media_type TEXT NOT NULL,
            picks_json TEXT NOT NULL DEFAULT '[]',
            message TEXT NOT NULL DEFAULT '',
            personalized INTEGER NOT NULL DEFAULT 1,
            ratings_count INTEGER NOT NULL DEFAULT 0,
            last_rated_at TIMESTAMP,
            ratings_revision INTEGER NOT NULL DEFAULT -1,
            requested_limit INTEGER NOT NULL DEFAULT 0,
            pipeline_version INTEGER NOT NULL DEFAULT 0,
            embedding_key TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, media_type)
        );
        """)

        # Ensure a default taste dossier row exists
        conn.execute("""
            INSERT OR IGNORE INTO taste_dossiers (user_id, full_summary, is_dirty)
            VALUES ('default_user', 'No ratings logged yet. Rate movies in the Rating Deck or Search to build your taste dossier!', 1)
        """)
        # Ensure deck refill cursor exists (single row, id=1)
        conn.execute("""
            INSERT OR IGNORE INTO deck_refill_state (id, next_page)
            VALUES (1, 1)
        """)

        # Migrate databases created by older releases.
        if "embedding_model" in _add_missing_columns(conn, "titles", {
            "imdb_id": "TEXT", "imdb_rating": "REAL", "embedding_model": "TEXT",
        }):
            conn.execute("DELETE FROM for_you_cache")
        _add_missing_columns(conn, "for_you_cache", {
            "last_rated_at": "TIMESTAMP",
            "ratings_revision": "INTEGER NOT NULL DEFAULT -1",
            "requested_limit": "INTEGER NOT NULL DEFAULT 0",
            "pipeline_version": "INTEGER NOT NULL DEFAULT 0",
            "embedding_key": "TEXT",
        })
        _add_missing_columns(conn, "taste_dossiers", {"evidence_version": "INTEGER NOT NULL DEFAULT 0"})
        _add_missing_columns(conn, "deck_refill_state", {"serve_count": "INTEGER NOT NULL DEFAULT 0"})
        _add_missing_columns(conn, "chat_messages", {"recommendation_reasons": "TEXT DEFAULT '{}'"})
        # Older releases created indexes that duplicate UNIQUE constraints.
        for index in ("idx_titles_tmdb", "idx_ratings_title", "idx_watchlist_title",
                      "idx_skipped_title", "idx_for_you_cache_user_media"):
            conn.execute(f"DROP INDEX IF EXISTS {index}")

        # A monotonic revision catches same-second edits and writes during synthesis.
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ratings_state (
                id INTEGER PRIMARY KEY CHECK(id = 1), revision INTEGER NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO ratings_state (id) VALUES (1);
        """)
        for event in ("INSERT", "UPDATE", "DELETE"):
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS ratings_changed_{event.lower()}
                AFTER {event} ON ratings BEGIN
                    UPDATE ratings_state SET revision = revision + 1 WHERE id = 1;
                    UPDATE taste_dossiers SET is_dirty = 1;
                    DELETE FROM for_you_cache;
                END
            """)

        logger.info("Database initialized successfully.")
