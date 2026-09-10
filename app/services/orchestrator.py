import asyncio
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
import numpy as np
from openai import AsyncOpenAI
from app.config import settings
from app.database import get_db, parse_utc_timestamp
from app.services.embeddings import embedding_service
from app.services.tmdb import tmdb_service
from app.services.taste_dossier import taste_dossier_service

logger = logging.getLogger(__name__)

# Adaptive taste drift constants (per Spec 0001)
HALF_LIFE_DAYS: float = 180.0
RECENCY_FLOOR: float = 0.25
RECENT_WINDOW_SIZE: int = 10
LONG_TERM_WEIGHT: float = 0.7
RECENT_WEIGHT: float = 0.3


@dataclass
class TasteVectorSignal:
    taste_vector: Optional[np.ndarray]
    liked_genres: Dict[str, int]
    liked_creators: Dict[str, int]
    liked_count: int
    loved_titles: List[Any]
    rating_rows: List[Any]
    ratings_count: int
    last_rated_at: Optional[str]


class OrchestratorService:
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = settings.OPENROUTER_API_KEY or "dummy_key_for_init"
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.OPENROUTER_BASE_URL
            )
        return self._client

    async def handle_vibe_query(
        self,
        query_text: str,
        session_id: str = "default_session",
        media_type_preference: str = "all"
    ) -> Dict[str, Any]:
        """
        Main Orchestrator pipeline:
        1. Extract semantic vibe and filters with GPT-4o.
        2. Vector search over local unrated SQLite titles.
        3. Fallback to TMDB discovery if candidates < 8.
        4. Re-rank and synthesize explanations with Taste Dossier via GPT-4o.
        """
        # Save user message to chat history
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'user', ?)",
                (session_id, query_text)
            )

        # Retrieve recent conversation history for context
        with get_db() as conn:
            history_rows = conn.execute("""
                SELECT role, content FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at DESC LIMIT 6
            """, (session_id,)).fetchall()
        chat_history = list(reversed([dict(r) for r in history_rows]))

        # Step 1: Parse query intent and extract filters
        intent = await self._parse_query_intent(query_text, media_type_preference)

        # Step 2: Local vector search
        candidates = await self._search_local_candidates(intent)

        # Step 3: Check candidate count and trigger dynamic TMDB discovery if needed
        if len(candidates) < 8 and settings.TMDB_API_KEY:
            logger.info(f"Local candidates count ({len(candidates)}) < 8. Triggering TMDB discovery fallback.")
            discovered = await self._discover_and_cache_tmdb(intent)
            if discovered:
                # Re-run search to include freshly embedded titles
                candidates = await self._search_local_candidates(intent)

        # Step 4: Retrieve Taste Dossier
        dossier = await taste_dossier_service.get_or_update_dossier()

        # Step 5: Final Reranking & Justification with GPT-4o
        final_result = await self._rerank_and_justify(
            query_text=query_text,
            chat_history=chat_history,
            candidates=candidates[:15],
            dossier=dossier
        )

        # Save assistant message & recommended title IDs
        rec_ids = [r["title_id"] for r in final_result.get("recommendations", [])]
        with get_db() as conn:
            conn.execute("""
                INSERT INTO chat_messages (session_id, role, content, recommended_title_ids)
                VALUES (?, 'assistant', ?, ?)
            """, (session_id, final_result.get("assistant_message", ""), json.dumps(rec_ids)))

        return {
            "assistant_message": final_result.get("assistant_message", ""),
            "recommendations": final_result.get("recommendations", []),
            "session_id": session_id
        }

    async def _parse_query_intent(self, query_text: str, media_type_pref: str) -> Dict[str, Any]:
        """Extract semantic vibe and structured filters from the user's prompt."""
        if not settings.OPENROUTER_API_KEY:
            return {
                "semantic_vibe": query_text,
                "media_type": media_type_pref if media_type_pref != "all" else None,
                "genres": [],
                "person": None,
                "year_min": None,
                "year_max": None
            }

        client = self._get_client()
        prompt = f"""
You are a movie and television search intent parser.
Analyze the user's request and output a JSON object to guide vector search and database filtering.

User Request: "{query_text}"
Selected Media Type Filter: "{media_type_pref}"

Return strictly a JSON object matching this schema:
{{
  "semantic_vibe": "A rich, descriptive paragraph describing the mood, narrative tropes, tone, aesthetic, and comparable themes to use for vector embedding search (e.g. 'nostalgic warm bittersweet romance with vibrant musicality and artistic ambition like La La Land')",
  "media_type": "movie" | "tv" | null,
  "genres": ["list of explicit TMDB genres if requested, e.g. Romance, Comedy, Sci-Fi"],
  "person": "name of director or actor if specifically requested, or null",
  "year_min": integer or null,
  "year_max": integer or null
}}
"""
        try:
            res = await client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": "You parse movie recommendations intent into JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            parsed = json.loads(res.choices[0].message.content)
            if media_type_pref in ("movie", "tv"):
                parsed["media_type"] = media_type_pref
            return parsed
        except Exception as e:
            logger.error(f"Error parsing query intent with GPT-4o: {e}")
            return {
                "semantic_vibe": query_text,
                "media_type": media_type_pref if media_type_pref != "all" else None,
                "genres": [],
                "person": None,
                "year_min": None,
                "year_max": None
            }

    async def _search_local_candidates(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform vector cosine similarity and SQL filtering over local unrated titles."""
        semantic_vibe = intent.get("semantic_vibe") or "great cinema"
        query_vec = await embedding_service.get_embedding(semantic_vibe)

        # Query unrated titles from SQLite
        with get_db() as conn:
            # Exclude already-rated titles and skipped titles
            query = """
                SELECT t.id, t.tmdb_id, t.media_type, t.title, t.release_year,
                       t.overview, t.poster_path, t.genres, t.director_or_creator,
                       t.cast_top, t.vote_average, t.popularity, t.embedding,
                       t.imdb_id, t.imdb_rating,
                       CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_on_watchlist
                FROM titles t
                LEFT JOIN ratings r ON t.id = r.title_id
                LEFT JOIN skipped_titles s ON t.id = s.title_id
                LEFT JOIN watchlist w ON t.id = w.title_id
                WHERE r.id IS NULL AND s.id IS NULL AND t.embedding IS NOT NULL
            """
            params: List[Any] = []

            if intent.get("media_type"):
                query += " AND t.media_type = ?"
                params.append(intent["media_type"])

            if intent.get("year_min"):
                query += " AND t.release_year >= ?"
                params.append(intent["year_min"])

            if intent.get("year_max"):
                query += " AND t.release_year <= ?"
                params.append(intent["year_max"])

            rows = conn.execute(query, params).fetchall()

        if not rows:
            return []

        doc_ids = []
        doc_rows = []
        vec_list = []

        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            v = embedding_service.bytes_to_vec(blob)
            if v.shape[0] == query_vec.shape[0]:
                vec_list.append(v)
                doc_ids.append(r["id"])
                doc_rows.append(r)

        if not vec_list:
            return []

        doc_matrix = np.vstack(vec_list)
        sim_scores = embedding_service.cosine_similarity(query_vec, doc_matrix)

        candidates = []
        for i, score in enumerate(sim_scores):
            row = doc_rows[i]
            # Prioritize watchlist items with slight boost (+0.08)
            effective_score = float(score) + (0.08 if row["is_on_watchlist"] else 0.0)
            candidates.append({
                "title_id": row["id"],
                "tmdb_id": row["tmdb_id"],
                "media_type": row["media_type"],
                "title": row["title"],
                "release_year": row["release_year"],
                "overview": row["overview"],
                "poster_path": row["poster_path"],
                "genres": json.loads(row["genres"] or "[]"),
                "director": row["director_or_creator"],
                "cast": json.loads(row["cast_top"] or "[]"),
                "vote_average": row["vote_average"],
                "imdb_id": row["imdb_id"] if "imdb_id" in row.keys() else None,
                "imdb_rating": (row["imdb_rating"] if "imdb_rating" in row.keys() and row["imdb_rating"] else row["vote_average"]) or 0.0,
                "is_on_watchlist": bool(row["is_on_watchlist"]),
                "similarity": effective_score
            })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates

    async def _discover_and_cache_tmdb(self, intent: Dict[str, Any]) -> List[int]:
        """Fetch candidates from TMDB discover, compute embeddings, and insert into SQLite concurrently."""
        media_type = intent.get("media_type") or "movie"
        genres = intent.get("genres") or []
        year_min = intent.get("year_min")
        year_max = intent.get("year_max")

        tmdb_items = await tmdb_service.discover_titles(
            media_type=media_type,
            year_min=year_min,
            year_max=year_max,
            limit=8
        )

        if not tmdb_items:
            return []

        sem = asyncio.Semaphore(4)

        async def _fetch_and_embed(item):
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
                    logger.error(f"Error processing discovery item {item.get('title')}: {e}")
                    return None

        results = await asyncio.gather(*(_fetch_and_embed(it) for it in tmdb_items[:8]))

        inserted_ids = []
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
                            popularity, imdb_id, imdb_rating, embedding, embedding_dim
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        full["tmdb_id"], full["media_type"], full["title"],
                        full.get("original_title"), full.get("release_year"),
                        full.get("overview"), full.get("poster_path"),
                        full.get("backdrop_path"), json.dumps(full.get("genres") or []),
                        full.get("director_or_creator"), json.dumps(full.get("cast_top") or []),
                        full.get("vote_average", 0.0), full.get("vote_count", 0),
                        full.get("popularity", 0.0), full.get("imdb_id"),
                        full.get("imdb_rating") or full.get("vote_average", 0.0),
                        blob, dim
                    ))
                    if cursor.lastrowid:
                        inserted_ids.append(cursor.lastrowid)
                except Exception as e:
                    logger.error(f"Error inserting discovery title {full.get('title')}: {e}")

        return inserted_ids

    def _build_taste_vector_and_stats(
        self,
        exclude_rating_ids: Optional[Set[int]] = None,
    ) -> TasteVectorSignal:
        """Builds a recency-weighted, dual-window blended taste vector and aggregate stats.

        Blends:
        1. Long-term vector: all ratings, weighted by score_weight * recency_factor
           (exponential decay with half-life of HALF_LIFE_DAYS, floored at RECENCY_FLOOR).
        2. Recent window vector: most recent RECENT_WINDOW_SIZE ratings, un-decayed.
        Final vector = normalize(0.7 * long_term + 0.3 * recent).
        """
        score_weights = {6: 2.0, 5: 1.5, 4: 1.0, 3: -0.2, 2: -1.0, 1: -1.5}
        now = datetime.now(timezone.utc)

        with get_db() as conn:
            query = """
                SELECT r.id as rating_id, r.score, r.aspect_tags, r.notes,
                       r.created_at, r.updated_at,
                       t.id, t.title, t.media_type, t.release_year, t.genres,
                       t.director_or_creator, t.embedding, t.vote_average, t.imdb_rating
                FROM ratings r
                JOIN titles t ON r.title_id = t.id
                ORDER BY COALESCE(r.updated_at, r.created_at) DESC
            """
            rating_rows = conn.execute(query).fetchall()

        if exclude_rating_ids:
            rating_rows = [r for r in rating_rows if r["rating_id"] not in exclude_rating_ids]

        liked_genres: Dict[str, int] = {}
        liked_creators: Dict[str, int] = {}
        liked_count = 0
        loved_titles: List[Any] = []
        last_rated_at: Optional[str] = None

        long_term_vecs = []
        recent_candidates = []

        for r in rating_rows:
            ts = r["updated_at"] or r["created_at"]
            if ts and (last_rated_at is None or str(ts) > str(last_rated_at)):
                last_rated_at = str(ts)

            score_val = int(r["score"] or 0)
            if score_val >= 4:
                liked_count += 1
                try:
                    for g in json.loads(r["genres"] or "[]"):
                        liked_genres[g] = liked_genres.get(g, 0) + 1
                except Exception:
                    pass
                if r["director_or_creator"]:
                    liked_creators[r["director_or_creator"]] = liked_creators.get(r["director_or_creator"], 0) + 1

            if score_val >= 5:
                loved_titles.append(r)

            blob = r["embedding"]
            if not blob:
                continue
            try:
                vec = embedding_service.bytes_to_vec(blob)
            except Exception:
                continue

            base_w = score_weights.get(score_val, 0.0)
            if base_w == 0:
                continue

            # Recency decay for long-term vector
            dt = parse_utc_timestamp(ts)
            if dt is None:
                recency_factor = RECENCY_FLOOR
            else:
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
                recency_factor = max(decay, RECENCY_FLOOR)

            long_term_vecs.append(vec * (base_w * recency_factor))
            recent_candidates.append((vec, base_w))

        # 1. Compute long-term vector
        long_term_vec: Optional[np.ndarray] = None
        if long_term_vecs:
            sum_long = np.sum(np.vstack(long_term_vecs), axis=0).astype(np.float32)
            norm_long = np.linalg.norm(sum_long)
            if norm_long > 0:
                long_term_vec = sum_long / norm_long

        # 2. Compute recent window vector (top RECENT_WINDOW_SIZE, un-decayed)
        recent_window = recent_candidates[:RECENT_WINDOW_SIZE]
        recent_vec: Optional[np.ndarray] = None
        if recent_window:
            recent_vecs = [v * w for v, w in recent_window]
            sum_recent = np.sum(np.vstack(recent_vecs), axis=0).astype(np.float32)
            norm_recent = np.linalg.norm(sum_recent)
            if norm_recent > 0:
                recent_vec = sum_recent / norm_recent

        # 3. Blend vectors
        if long_term_vec is None and recent_vec is None:
            final_vec = None
        elif recent_vec is None:
            final_vec = long_term_vec
        elif long_term_vec is None:
            final_vec = recent_vec
        else:
            blended = (LONG_TERM_WEIGHT * long_term_vec + RECENT_WEIGHT * recent_vec).astype(np.float32)
            norm_b = np.linalg.norm(blended)
            final_vec = (blended / norm_b).astype(np.float32) if norm_b > 0 else long_term_vec

        return TasteVectorSignal(
            taste_vector=final_vec,
            liked_genres=liked_genres,
            liked_creators=liked_creators,
            liked_count=liked_count,
            loved_titles=loved_titles,
            rating_rows=rating_rows,
            ratings_count=len(rating_rows),
            last_rated_at=last_rated_at,
        )

    async def get_personalized_picks(
        self,
        limit: int = 5,
        media_type_preference: str = "all",
        user_id: str = "default_user",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Suggest `limit` Titles most likely to be liked, grounded in Ratings + Taste Dossier.

        Pipeline:
        1. Check for_you_cache; if cached picks exist, force_refresh is False, and ratings
           have not changed (count and last_rated_at match), re-hydrate dynamic user state
           (watchlist, ratings) and return immediately.
        2. If dirty or refreshed, build blended taste vector via `_build_taste_vector_and_stats`
           combining exponential decay on long-term ratings and recent window ratings.
        3. Vector-search unrated Titles (rated Titles excluded; skipped Titles
           included since Deck skip means "haven't seen", not "disliked").
        4. Rerank top candidates against the Taste Dossier via GPT-4o when
           configured, otherwise use a deterministic similarity + quality blend.
        5. Persist the generated shelf into for_you_cache along with ratings_count and last_rated_at.
        """
        limit = max(1, min(limit, 10))
        normalized_media = media_type_preference if media_type_preference in ("movie", "tv") else "movie"
        media_filter = media_type_preference if media_type_preference in ("movie", "tv") else None

        if not force_refresh:
            with get_db() as conn:
                cached_row = conn.execute("""
                    SELECT picks_json, message, personalized, ratings_count, last_rated_at
                    FROM for_you_cache
                    WHERE user_id = ? AND media_type = ?
                """, (user_id, normalized_media)).fetchone()
                current_ratings_count = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
                current_last_rated_at = conn.execute(
                    "SELECT MAX(COALESCE(updated_at, created_at)) FROM ratings"
                ).fetchone()[0]

            if cached_row:
                was_personalized = bool(cached_row["personalized"])
                cached_count = cached_row["ratings_count"] if cached_row["ratings_count"] is not None else 0
                cached_last_rated = cached_row["last_rated_at"]

                # Auto-refresh if:
                # 1. Previously cold-start and user now has rated at least one Title
                # 2. Ratings count changed (new rating or deletion)
                # 3. Maximum rating timestamp changed (new rating or re-rating)
                should_auto_refresh = (
                    (not was_personalized and current_ratings_count > 0)
                    or (cached_count != current_ratings_count)
                    or (cached_last_rated != current_last_rated_at)
                )

                if not should_auto_refresh:
                    try:
                        cached_picks = json.loads(cached_row["picks_json"] or "[]")
                    except Exception:
                        cached_picks = []

                    # Invalidate cache if any pick contains obsolete vector jargon or generic placeholder
                    if cached_picks and any(
                        "vector" in str(p.get("reason", "")).lower()
                        or "taste profile" in str(p.get("reason", "")).lower()
                        or not p.get("reason")
                        for p in cached_picks
                    ):
                        should_auto_refresh = True

                    if not should_auto_refresh and cached_picks:
                        title_ids = [p["title_id"] for p in cached_picks if isinstance(p, dict) and "title_id" in p]
                        with get_db() as conn:
                            if title_ids:
                                placeholders = ",".join("?" * len(title_ids))
                                wl_rows = conn.execute(
                                    f"SELECT title_id FROM watchlist WHERE title_id IN ({placeholders})",
                                    title_ids
                                ).fetchall()
                                r_rows = conn.execute(
                                    f"SELECT title_id, score FROM ratings WHERE title_id IN ({placeholders})",
                                    title_ids
                                ).fetchall()
                                wl_set = {r[0] for r in wl_rows}
                                r_map = {r[0]: r[1] for r in r_rows}
                            else:
                                wl_set = set()
                                r_map = {}

                        for p in cached_picks:
                            tid = p.get("title_id")
                            p["is_on_watchlist"] = tid in wl_set
                            p["user_rating"] = r_map.get(tid)

                        dossier = await taste_dossier_service.get_or_update_dossier(user_id=user_id)
                        return {
                            "picks": cached_picks[:limit],
                            "dossier": dossier,
                            "ratings_count": current_ratings_count,
                            "liked_count": 0,
                            "message": cached_row["message"],
                            "personalized": was_personalized,
                        }

        signal = self._build_taste_vector_and_stats()
        rating_rows = signal.rating_rows
        ratings_count = signal.ratings_count
        liked_count = signal.liked_count
        liked_genres = signal.liked_genres
        liked_creators = signal.liked_creators
        taste_vec = signal.taste_vector
        last_rated_at = signal.last_rated_at

        dossier = await taste_dossier_service.get_or_update_dossier(user_id=user_id)

        if not rating_rows:
            # Cold start: no taste signal — fall back to popular unrated Titles.
            fallback = await self._popular_unrated_fallback(limit, media_filter)
            message = (
                "Rate a few Titles and I'll tailor this shelf to you. "
                "Showing popular crowd-pleasers for now."
            )
            self._save_for_you_cache(user_id, normalized_media, fallback, message, False, 0, None)
            return {
                "picks": fallback,
                "dossier": dossier,
                "ratings_count": 0,
                "liked_count": 0,
                "message": message,
                "personalized": False,
            }

        if taste_vec is None:
            fallback = await self._popular_unrated_fallback(limit, media_filter)
            message = "Couldn't build a taste signal from your Ratings yet — showing popular Titles."
            self._save_for_you_cache(user_id, normalized_media, fallback, message, False, ratings_count, last_rated_at)
            return {
                "picks": fallback,
                "dossier": dossier,
                "ratings_count": ratings_count,
                "liked_count": liked_count,
                "message": message,
                "personalized": False,
            }

        # --- 2. Score unrated Titles against the taste vector ---
        with get_db() as conn:
            query = """
                SELECT t.id, t.tmdb_id, t.media_type, t.title, t.release_year,
                       t.overview, t.poster_path, t.genres, t.director_or_creator,
                       t.cast_top, t.vote_average, t.vote_count, t.popularity,
                       t.embedding, t.imdb_id, t.imdb_rating,
                       CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_on_watchlist
                FROM titles t
                LEFT JOIN ratings r ON t.id = r.title_id
                LEFT JOIN watchlist w ON t.id = w.title_id
                WHERE r.id IS NULL AND t.embedding IS NOT NULL
            """
            params: List[Any] = []
            if media_filter:
                query += " AND t.media_type = ?"
                params.append(media_filter)
            rows = conn.execute(query, params).fetchall()

        scored: List[Dict[str, Any]] = []
        for row in rows:
            blob = row["embedding"]
            if not blob:
                continue
            try:
                v = embedding_service.bytes_to_vec(blob)
            except Exception:
                continue
            if v.shape[0] != taste_vec.shape[0]:
                continue
            sim = float(np.dot(v, taste_vec))
            # Small quality prior so ties break toward well-regarded Titles,
            # plus a watchlist boost for Titles the user already flagged.
            quality = min((row["vote_average"] or 0.0) / 10.0, 1.0) * 0.05
            quality += min((row["popularity"] or 0.0) / 200.0, 1.0) * 0.03
            watchlist_boost = 0.08 if row["is_on_watchlist"] else 0.0
            scored.append({
                "title_id": row["id"],
                "tmdb_id": row["tmdb_id"],
                "media_type": row["media_type"],
                "title": row["title"],
                "release_year": row["release_year"],
                "overview": row["overview"],
                "poster_path": row["poster_path"],
                "genres": json.loads(row["genres"] or "[]"),
                "director": row["director_or_creator"],
                "cast": json.loads(row["cast_top"] or "[]"),
                "vote_average": row["vote_average"],
                "vote_count": row["vote_count"],
                "popularity": row["popularity"],
                "imdb_id": row["imdb_id"] if "imdb_id" in row.keys() else None,
                "imdb_rating": (row["imdb_rating"] if "imdb_rating" in row.keys() and row["imdb_rating"] else row["vote_average"]) or 0.0,
                "is_on_watchlist": bool(row["is_on_watchlist"]),
                "similarity": sim + quality + watchlist_boost,
                "raw_similarity": sim,
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        shortlist = scored[:15]

        if not shortlist:
            return {
                "picks": [],
                "dossier": dossier,
                "ratings_count": ratings_count,
                "liked_count": liked_count,
                "message": "You've rated everything in the catalog — add more Titles via Search to keep picks coming.",
                "personalized": True,
            }

        # Select loved titles matching the requested media type first
        media_loved = [
            r for r in signal.loved_titles
            if r["media_type"] == normalized_media
        ]
        if not media_loved:
            # Fall back to loved titles from any media type if none exist for this type
            media_loved = signal.loved_titles

        # --- 3. Rerank against the Taste Dossier (LLM when available) ---
        if settings.OPENROUTER_API_KEY:
            picks = await self._rerank_personalized_picks(
                shortlist,
                dossier,
                ratings_count,
                limit,
                liked_genres=liked_genres,
                liked_creators=liked_creators,
                loved_titles=media_loved,
            )
        else:
            picks = self._heuristic_personalized_reasons(
                shortlist[:limit],
                liked_genres=liked_genres,
                liked_creators=liked_creators,
                loved_titles=media_loved,
                dossier=dossier,
            )

        # Prioritize loved titles that share genres with the recommended picks
        pick_genres = set()
        for p in picks:
            for g in p.get("genres", []):
                pick_genres.add(g)

        def loved_sort_key(r):
            score = int(r["score"] or 0)
            overlap = 0
            if pick_genres and r["genres"]:
                try:
                    g_list = json.loads(r["genres"]) if isinstance(r["genres"], str) else r["genres"]
                    overlap = sum(1 for g in g_list if g in pick_genres)
                except Exception:
                    pass
            return (score, overlap)

        sorted_loved = sorted(media_loved, key=loved_sort_key, reverse=True)

        # On force_refresh, sample/rotate among top-scoring candidates if more than 5 exist
        top_score = int(sorted_loved[0]["score"] or 0) if sorted_loved else 0
        top_tier = [r for r in sorted_loved if int(r["score"] or 0) == top_score]
        if force_refresh and len(top_tier) > 5:
            chosen = random.sample(top_tier, 5)
            chosen.sort(key=loved_sort_key, reverse=True)
        else:
            chosen = sorted_loved[:5]

        loved_titles = [
            f"'{r['title']}' ({r['score']}/6)"
            for r in chosen
        ]
        if loved_titles:
            message = f"Based on your love for {', '.join(loved_titles)} — top {len(picks)} picks from your Dossier."
        else:
            message = f"Based on your {ratings_count} Ratings — top {len(picks)} picks from your Dossier."

        self._save_for_you_cache(user_id, normalized_media, picks, message, True, ratings_count, last_rated_at)

        return {
            "picks": picks,
            "dossier": dossier,
            "ratings_count": ratings_count,
            "liked_count": liked_count,
            "message": message,
            "personalized": True,
        }

    def _save_for_you_cache(
        self,
        user_id: str,
        media_type: str,
        picks: List[Dict[str, Any]],
        message: str,
        personalized: bool,
        ratings_count: int,
        last_rated_at: Optional[str] = None,
    ) -> None:
        """Persist generated for-you recommendation shelf to SQLite cache."""
        try:
            with get_db() as conn:
                conn.execute("""
                    INSERT INTO for_you_cache (
                        user_id, media_type, picks_json, message, personalized,
                        ratings_count, last_rated_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, media_type) DO UPDATE SET
                        picks_json = excluded.picks_json,
                        message = excluded.message,
                        personalized = excluded.personalized,
                        ratings_count = excluded.ratings_count,
                        last_rated_at = excluded.last_rated_at,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id,
                    media_type,
                    json.dumps(picks),
                    message,
                    1 if personalized else 0,
                    ratings_count,
                    last_rated_at,
                ))
        except Exception as e:
            logger.error(f"Failed to cache for-you picks: {e}")

    async def _popular_unrated_fallback(
        self, limit: int, media_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Top popular unrated Titles for cold-start users."""
        with get_db() as conn:
            query = """
                SELECT t.id, t.tmdb_id, t.media_type, t.title, t.release_year,
                       t.overview, t.poster_path, t.genres, t.director_or_creator,
                       t.cast_top, t.vote_average, t.vote_count, t.popularity,
                       t.imdb_id, t.imdb_rating,
                       CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_on_watchlist
                FROM titles t
                LEFT JOIN ratings r ON t.id = r.title_id
                LEFT JOIN watchlist w ON t.id = w.title_id
                WHERE r.id IS NULL
            """
            params: List[Any] = []
            if media_filter:
                query += " AND t.media_type = ?"
                params.append(media_filter)
            query += " ORDER BY t.popularity DESC, t.vote_count DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()

        picks = []
        for row in rows:
            picks.append({
                "title_id": row["id"],
                "tmdb_id": row["tmdb_id"],
                "media_type": row["media_type"],
                "title": row["title"],
                "release_year": row["release_year"],
                "overview": row["overview"],
                "poster_path": row["poster_path"],
                "genres": json.loads(row["genres"] or "[]"),
                "director": row["director_or_creator"],
                "cast": json.loads(row["cast_top"] or "[]"),
                "vote_average": row["vote_average"],
                "imdb_rating": (row["imdb_rating"] if "imdb_rating" in row.keys() and row["imdb_rating"] else row["vote_average"]) or 0.0,
                "imdb_id": row["imdb_id"] if "imdb_id" in row.keys() else None,
                "is_on_watchlist": bool(row["is_on_watchlist"]),
                "similarity": 0.0,
                "reason": "A popular crowd-pleaser to start your taste journey — rate it to sharpen future picks.",
            })
        return picks

    def _generate_personalized_reason(
        self,
        candidate: Dict[str, Any],
        liked_genres: Optional[Dict[str, int]] = None,
        liked_creators: Optional[Dict[str, int]] = None,
        loved_titles: Optional[List[Any]] = None,
        dossier: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Constructs a film-literate, compelling explanation of WHY this Title is recommended.

        Prioritizes:
        1. Creator/director affinity matching user's high ratings.
        2. Strong genre, tone, and emotional resonance with specific loved titles.
        3. High affinity for user's favorite genres with critical acclaim or character focus.
        4. Atmospheric or narrative preferences from the user's taste dossier.
        5. Backlog relevance if already saved on the user's watchlist.
        6. Film-literate aesthetic/trope summary grounded in the title's attributes.
        Never exposes technical jargon (e.g. 'vector', 'similarity score').
        """
        c_title = candidate.get("title", "")
        c_genres = candidate.get("genres") or []
        if isinstance(c_genres, str):
            try:
                c_genres = json.loads(c_genres)
            except Exception:
                c_genres = [c_genres]

        c_director = candidate.get("director") or candidate.get("director_or_creator")
        c_score = float(candidate.get("imdb_rating") or candidate.get("vote_average") or 0.0)
        is_wl = bool(candidate.get("is_on_watchlist", False))

        # 1. Creator affinity
        if c_director and liked_creators and liked_creators.get(c_director, 0) > 0:
            if is_wl:
                return f"From director {c_director}, a filmmaker you love, waiting on your Watchlist."
            return f"From director {c_director}, whose storytelling and visual craft you've rated highly."

        # 2. Connections with specific user loved titles (rated 5 or 6)
        if loved_titles:
            best_overlap_loved = None
            best_overlap_count = 0
            best_shared_genres = []
            for loved in loved_titles:
                l_genres = loved.get("genres") if isinstance(loved, dict) else loved["genres"]
                if isinstance(l_genres, str):
                    try:
                        l_genres = json.loads(l_genres)
                    except Exception:
                        l_genres = []
                elif not isinstance(l_genres, list):
                    l_genres = []
                shared = [g for g in c_genres if g in l_genres]
                if len(shared) > best_overlap_count:
                    best_overlap_count = len(shared)
                    best_overlap_loved = loved
                    best_shared_genres = shared

            if best_overlap_loved and best_overlap_count >= 2:
                loved_name = best_overlap_loved.get("title") if isinstance(best_overlap_loved, dict) else best_overlap_loved["title"]
                shared_str = " & ".join(best_shared_genres[:2])
                if is_wl:
                    return f"Shares the {shared_str} tone and emotional resonance of your favorite '{loved_name}', waiting on your Watchlist."
                return f"Shares the rich {shared_str} storytelling and emotional resonance you loved in '{loved_name}'."
            elif best_overlap_loved and best_overlap_count == 1:
                loved_name = best_overlap_loved.get("title") if isinstance(best_overlap_loved, dict) else best_overlap_loved["title"]
                shared_genre = best_shared_genres[0]
                if is_wl:
                    return f"Recommended because you loved '{loved_name}', offering a compelling {shared_genre} experience from your Watchlist."
                return f"Recommended because you loved '{loved_name}', capturing a similarly compelling {shared_genre} narrative."

        # 3. Genre affinity match
        if liked_genres:
            matching_genres = [g for g in c_genres if g in liked_genres]
            if matching_genres:
                matching_genres.sort(key=lambda g: liked_genres.get(g, 0), reverse=True)
                genre_str = " & ".join(matching_genres[:2])
                if is_wl:
                    return f"A standout {genre_str} on your Watchlist that aligns closely with your viewing habits."
                if c_score >= 7.5:
                    return f"An acclaimed {genre_str} ({c_score:.1f}/10) celebrated for its compelling craft and narrative depth."
                return f"Tailored for your love of {genre_str}, offering strong character depth and engaging pacing."

        # 4. Dossier atmosphere / core love match
        if dossier:
            atmospheres = dossier.get("atmospheric_preferences") or []
            core_loves = dossier.get("core_loves") or []
            if atmospheres:
                return f"Aligns with your preference for {atmospheres[0].lower()} storytelling, featuring immersive craft and performances."
            if core_loves:
                return f"Echoes your affinity for {core_loves[0].lower()} with its evocative narrative and strong direction."

        # 5. Natural fallback using title attributes
        if is_wl:
            if c_genres:
                return f"A compelling {' & '.join(c_genres[:2])} waiting on your Watchlist, ready for your next watch."
            return "A standout title waiting on your Watchlist, curated for your viewing journey."

        if c_genres:
            genre_str = " & ".join(c_genres[:2])
            if c_score >= 7.5:
                return f"A critically acclaimed {genre_str} celebrated for its rich world-building and memorable performances."
            return f"A crowd-pleasing {genre_str} known for its engaging story and strong lead performances."

        return "A critically acclaimed selection recognized for its immersive storytelling and standout craft."

    def _heuristic_personalized_reasons(
        self,
        candidates: List[Dict[str, Any]],
        liked_genres: Dict[str, int],
        liked_creators: Dict[str, int],
        loved_titles: Optional[List[Any]] = None,
        dossier: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Explain each pick via genre/creator overlap and connection to user favorites."""
        picks = []
        for c in candidates:
            reason = self._generate_personalized_reason(
                c, liked_genres=liked_genres, liked_creators=liked_creators, loved_titles=loved_titles, dossier=dossier
            )
            picks.append({**c, "reason": reason})
        return picks

    async def _rerank_personalized_picks(
        self,
        shortlist: List[Dict[str, Any]],
        dossier: Dict[str, Any],
        ratings_count: int,
        limit: int,
        liked_genres: Optional[Dict[str, int]] = None,
        liked_creators: Optional[Dict[str, int]] = None,
        loved_titles: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Use GPT-4o to pick the best `limit` Titles and justify each from the Dossier."""
        client = self._get_client()
        lines = []
        for i, c in enumerate(shortlist):
            wl = " [ALREADY ON USER WATCHLIST]" if c.get("is_on_watchlist") else ""
            lines.append(
                f"Candidate #{i+1} [ID: {c['title_id']}]: '{c['title']}' "
                f"({c['release_year']}, {c['media_type']}){wl}\n"
                f"  Genres: {', '.join(c['genres'])} | Director: {c['director'] or 'N/A'} | "
                f"Rating: {c.get('imdb_rating') or c.get('vote_average', 0.0):.1f}/10\n"
                f"  Synopsis: {c['overview']}\n"
            )

        loved_summary = []
        if loved_titles:
            for r in loved_titles[:6]:
                r_title = r.get("title") if isinstance(r, dict) else r["title"]
                r_year = r.get("release_year") if isinstance(r, dict) else r["release_year"]
                r_media = r.get("media_type") if isinstance(r, dict) else r["media_type"]
                r_score = r.get("score") if isinstance(r, dict) else r["score"]
                loved_summary.append(f"'{r_title}' ({r_year or '—'}, {r_media}) - Rated {r_score}/6")

        prompt = f"""
You are CineMatch, an insightful film scholar picking Titles the user will most likely love.
Base your choice ONLY on the candidate list and the user's Taste Dossier + Rating history.

### Taste Dossier (from {ratings_count} Ratings):
Core Loves: {json.dumps(dossier.get('core_loves', []))}
Deal Breakers: {json.dumps(dossier.get('deal_breakers', []))}
Creator Affinities: {json.dumps(dossier.get('creator_affinities', []))}
Atmospheric Preferences: {json.dumps(dossier.get('atmospheric_preferences', []))}
Taste Summary: {dossier.get('full_summary', 'N/A')}

### User's Top Rated Favorites:
{json.dumps(loved_summary)}

### Candidate Titles:
{chr(10).join(lines)}

### Instructions:
1. Select EXACTLY {limit} titles from the candidates (no more, no fewer) that the user is most likely to love.
2. Prefer candidates aligning with loved genres, directors, and tropes; demote anything clashing with deal-breakers.
3. CRITICAL: For EVERY selected title without exception, provide a personalized 1-2 sentence justification in 'reason' explaining specifically WHY the user will love it. Cite specific story elements, aesthetic, tone, director style, or connections to their loved titles. NEVER use generic placeholder phrases like "matches your taste" or technical jargon like "vector similarity".
4. Write a warm one-line intro message for the shelf.

Respond ONLY in valid JSON matching this schema:
{{
  "intro": "...",
  "picks": [
    {{
      "title_id": <int>,
      "reason": "<1-2 sentence customized explanation stating why>"
    }}
  ]
}}
"""
        try:
            res = await client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": "You output strictly valid JSON for movie recommendations."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            data = json.loads(res.choices[0].message.content)
            by_id = {c["title_id"]: c for c in shortlist}
            picks = []
            for entry in data.get("picks", [])[:limit]:
                tid = entry.get("title_id")
                if tid in by_id:
                    reason = str(entry.get("reason", "")).strip()
                    # If reason is too short, generic, or contains vector jargon, generate a proper one
                    if (
                        not reason
                        or len(reason) < 15
                        or "vector" in reason.lower()
                        or "matches your taste" in reason.lower()
                        or "taste profile" in reason.lower()
                    ):
                        reason = self._generate_personalized_reason(
                            by_id[tid], liked_genres, liked_creators, loved_titles, dossier
                        )
                    picks.append({**by_id[tid], "reason": reason})

            # Fill short if the model returned fewer than requested.
            if len(picks) < limit:
                seen = {p["title_id"] for p in picks}
                for c in shortlist:
                    if c["title_id"] not in seen:
                        backfill_reason = c.get("reason")
                        if (
                            not backfill_reason
                            or "vector" in str(backfill_reason).lower()
                            or "matches your taste" in str(backfill_reason).lower()
                            or "taste profile" in str(backfill_reason).lower()
                        ):
                            backfill_reason = self._generate_personalized_reason(
                                c, liked_genres, liked_creators, loved_titles, dossier
                            )
                        picks.append({**c, "reason": backfill_reason})
                    if len(picks) >= limit:
                        break
            return picks[:limit]
        except Exception as e:
            logger.error(f"Error reranking personalized picks with GPT-4o: {e}")
            return self._heuristic_personalized_reasons(
                shortlist[:limit], liked_genres or {}, liked_creators or {}, loved_titles=loved_titles, dossier=dossier
            )

    async def _rerank_and_justify(
        self,
        query_text: str,
        chat_history: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        dossier: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use GPT-4o to evaluate candidates against the user's taste dossier and write tailored justifications."""
        if not candidates:
            return {
                "assistant_message": "I searched the catalog and TMDB, but couldn't find unwatched titles matching those exact criteria. Try broadening your vibe or asking for a different genre!",
                "recommendations": []
            }

        if not settings.OPENROUTER_API_KEY:
            # Offline / dummy fallback
            top_3 = candidates[:3]
            recs = []
            for c in top_3:
                reason = self._generate_personalized_reason(c, dossier=dossier)
                badge = " (Already on your Watchlist!)" if c.get("is_on_watchlist") else ""
                recs.append({
                    "title_id": c["title_id"],
                    "tmdb_id": c["tmdb_id"],
                    "media_type": c["media_type"],
                    "title": c["title"],
                    "release_year": c["release_year"],
                    "poster_path": c["poster_path"],
                    "genres": c["genres"],
                    "overview": c["overview"],
                    "reason": f"{reason}{badge}",
                    "is_on_watchlist": c.get("is_on_watchlist", False),
                    "vote_average": c.get("vote_average", 0.0),
                    "imdb_rating": c.get("imdb_rating") or c.get("vote_average", 0.0),
                    "imdb_id": c.get("imdb_id"),
                })
            return {
                "assistant_message": f"Here are {len(recs)} top recommendations curated for your taste tonight:",
                "recommendations": recs
            }

        client = self._get_client()

        candidates_summary = []
        for i, c in enumerate(candidates):
            wl_str = " [ALREADY ON USER WATCHLIST]" if c.get("is_on_watchlist") else ""
            candidates_summary.append(
                f"Candidate #{i+1} [ID: {c['title_id']}]: '{c['title']}' ({c['release_year']}, {c['media_type']}){wl_str}\n"
                f"  Genres: {', '.join(c['genres'])} | Director: {c['director'] or 'N/A'}\n"
                f"  Synopsis: {c['overview']}\n"
            )

        prompt = f"""
You are CineMatch, an insightful, warm, and hyper-literate film scholar and recommendation concierge.
The user is asking for movie or series recommendations.

### User Request:
"{query_text}"

### User's Taste Dossier:
Core Loves: {json.dumps(dossier.get('core_loves', []))}
Deal Breakers: {json.dumps(dossier.get('deal_breakers', []))}
Creator Affinities: {json.dumps(dossier.get('creator_affinities', []))}
Atmospheric Preferences: {json.dumps(dossier.get('atmospheric_preferences', []))}
Taste Summary: {dossier.get('full_summary', 'N/A')}

### Candidate Titles Retrieved:
{chr(10).join(candidates_summary)}

### Instructions:
1. Select the BEST 3 to 5 titles from the candidate list that fit both the requested vibe AND respect the user's Taste Dossier (avoiding their deal-breakers, honoring their loved tropes).
2. If any chosen candidate has [ALREADY ON USER WATCHLIST], highlight it as an exciting discovery from their own backlog!
3. For each recommended title, write a 1-2 sentence compelling personalized justification explaining *why* it fits their taste (referencing aesthetic, tone, pacing, or storytelling specifics).
4. Write a warm, cinephile conversational message summarizing why this selection was curated for tonight.

Respond ONLY in valid JSON matching this schema:
{{
  "assistant_message": "Conversational message to the user introducing the suggestions...",
  "recommendations": [
    {{
      "title_id": integer (must match one of the candidate IDs provided above),
      "reason": "1-2 sentence customized justification connecting to user taste and query"
    }}
  ]
}}
"""

        try:
            res = await client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": "You are CineMatch, a conversational film concierge who outputs strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4
            )
            data = json.loads(res.choices[0].message.content)
            cand_by_id = {c["title_id"]: c for c in candidates}

            formatted_recs = []
            for r in data.get("recommendations", []):
                tid = r.get("title_id")
                if tid in cand_by_id:
                    cand = cand_by_id[tid]
                    reason = str(r.get("reason", "")).strip()
                    if (
                        not reason
                        or len(reason) < 15
                        or "vector" in reason.lower()
                        or "preferences" in reason.lower()
                    ):
                        reason = self._generate_personalized_reason(cand, dossier=dossier)
                    formatted_recs.append({
                        "title_id": tid,
                        "tmdb_id": cand["tmdb_id"],
                        "media_type": cand["media_type"],
                        "title": cand["title"],
                        "release_year": cand["release_year"],
                        "poster_path": cand["poster_path"],
                        "genres": cand["genres"],
                        "overview": cand["overview"],
                        "reason": reason,
                        "is_on_watchlist": cand.get("is_on_watchlist", False),
                        "vote_average": cand.get("vote_average", 0.0),
                        "imdb_rating": cand.get("imdb_rating") or cand.get("vote_average", 0.0),
                        "imdb_id": cand.get("imdb_id"),
                    })

            return {
                "assistant_message": data.get("assistant_message", "Here are my top recommendations for you tonight:"),
                "recommendations": formatted_recs
            }
        except Exception as e:
            logger.error(f"Error in reranker GPT-4o call: {e}")
            top_3 = candidates[:3]
            recs = [{
                "title_id": c["title_id"],
                "tmdb_id": c["tmdb_id"],
                "media_type": c["media_type"],
                "title": c["title"],
                "release_year": c["release_year"],
                "poster_path": c["poster_path"],
                "genres": c["genres"],
                "overview": c["overview"],
                "reason": self._generate_personalized_reason(c, dossier=dossier),
                "is_on_watchlist": c.get("is_on_watchlist", False),
                "vote_average": c.get("vote_average", 0.0),
                "imdb_rating": c.get("imdb_rating") or c.get("vote_average", 0.0),
                "imdb_id": c.get("imdb_id"),
            } for c in top_3]
            return {
                "assistant_message": "Here are curated recommendations based on your request:",
                "recommendations": recs
            }

    async def get_deck_suggestion(self, exclude_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """Returns a single top unseen candidate personalized to the user's taste."""
        signal = self._build_taste_vector_and_stats()
        if signal.ratings_count < 2:
            return None

        taste_vec = signal.taste_vector
        if taste_vec is None:
            return None

        liked_genres = signal.liked_genres
        liked_creators = signal.liked_creators

        with get_db() as conn:
            rows = conn.execute("""
                SELECT t.*
                FROM titles t
                LEFT JOIN ratings r ON t.id = r.title_id
                LEFT JOIN skipped_titles s ON t.id = s.title_id
                LEFT JOIN watchlist w ON t.id = w.title_id
                WHERE r.id IS NULL AND s.id IS NULL AND w.id IS NULL
                  AND t.embedding IS NOT NULL
            """).fetchall()

        if exclude_ids:
            rows = [r for r in rows if r["id"] not in exclude_ids]

        if not rows:
            return None

        valid_rows = []
        vec_list = []
        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            try:
                v = embedding_service.bytes_to_vec(blob)
                if v.shape[0] == taste_vec.shape[0]:
                    vec_list.append(v)
                    valid_rows.append(r)
            except Exception:
                continue

        if not vec_list:
            return None

        doc_matrix = np.vstack(vec_list)
        sim_scores = embedding_service.cosine_similarity(taste_vec, doc_matrix)

        scored = []
        for i, raw_sim in enumerate(sim_scores):
            r = valid_rows[i]
            sim = float(raw_sim)
            quality = min((r["vote_average"] or 0.0) / 10.0, 1.0) * 0.05
            eff_score = sim + quality
            scored.append((eff_score, sim, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_eff, best_sim, best_row = scored[0]

        if best_sim < 0.35:
            return None

        genres = json.loads(best_row["genres"] or "[]")
        director = best_row["director_or_creator"]

        matching_genres = [g for g in genres if g in liked_genres]
        if director and director in liked_creators:
            reason = f"Matches your affinity for {director}"
        elif matching_genres:
            reason = f"Matches your taste in {' & '.join(matching_genres[:2])}"
        else:
            reason = f"Tailored to your taste, offering a compelling {genres[0] if genres else 'film'} experience."

        match_score = min(98, max(75, int(best_sim * 100)))

        return {
            "id": best_row["id"],
            "tmdb_id": best_row["tmdb_id"],
            "media_type": best_row["media_type"],
            "title": best_row["title"],
            "original_title": best_row["original_title"],
            "release_year": best_row["release_year"],
            "overview": best_row["overview"],
            "poster_path": best_row["poster_path"],
            "backdrop_path": best_row["backdrop_path"],
            "genres": genres,
            "director_or_creator": director,
            "cast_top": json.loads(best_row["cast_top"] or "[]"),
            "vote_average": best_row["vote_average"],
            "vote_count": best_row["vote_count"],
            "popularity": best_row["popularity"],
            "is_suggestion": True,
            "suggestion_reason": reason,
            "match_score": match_score,
            "is_in_watchlist": False,
        }

orchestrator_service = OrchestratorService()
