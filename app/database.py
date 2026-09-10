import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from app.config import settings

logger = logging.getLogger(__name__)

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
    except Exception:
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

def init_db():
    """Create tables and indexes if they do not exist."""
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tmdb_id, media_type)
        );

        CREATE INDEX IF NOT EXISTS idx_titles_tmdb ON titles(tmdb_id, media_type);
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

        CREATE INDEX IF NOT EXISTS idx_ratings_title ON ratings(title_id);
        CREATE INDEX IF NOT EXISTS idx_ratings_score ON ratings(score);

        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title_id INTEGER NOT NULL REFERENCES titles(id) ON DELETE CASCADE,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title_id)
        );

        CREATE INDEX IF NOT EXISTS idx_watchlist_title ON watchlist(title_id);

        CREATE TABLE IF NOT EXISTS skipped_titles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title_id INTEGER NOT NULL REFERENCES titles(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title_id)
        );

        CREATE INDEX IF NOT EXISTS idx_skipped_title ON skipped_titles(title_id);

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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id)
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL DEFAULT 'default_session',
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            recommended_title_ids TEXT DEFAULT '[]', -- JSON list of title_ids
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, created_at);

        CREATE TABLE IF NOT EXISTS deck_refill_state (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            next_page INTEGER NOT NULL DEFAULT 1
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, media_type)
        );

        CREATE INDEX IF NOT EXISTS idx_for_you_cache_user_media ON for_you_cache(user_id, media_type);
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

        # Safely migrate titles table for existing databases
        existing_cols = [c[1] for c in conn.execute("PRAGMA table_info(titles)").fetchall()]
        if "imdb_id" not in existing_cols:
            conn.execute("ALTER TABLE titles ADD COLUMN imdb_id TEXT")
        if "imdb_rating" not in existing_cols:
            conn.execute("ALTER TABLE titles ADD COLUMN imdb_rating REAL")
        if "embedding_model" not in existing_cols:
            conn.execute("ALTER TABLE titles ADD COLUMN embedding_model TEXT")
            conn.execute("DELETE FROM for_you_cache")

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

        # Safely migrate for_you_cache table for existing databases
        existing_cache_cols = [c[1] for c in conn.execute("PRAGMA table_info(for_you_cache)").fetchall()]
        if "last_rated_at" not in existing_cache_cols:
            conn.execute("ALTER TABLE for_you_cache ADD COLUMN last_rated_at TIMESTAMP")
        if "ratings_revision" not in existing_cache_cols:
            conn.execute("ALTER TABLE for_you_cache ADD COLUMN ratings_revision INTEGER NOT NULL DEFAULT -1")

        logger.info("Database initialized successfully.")
