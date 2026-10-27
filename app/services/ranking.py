"""Score unrated Titles against the user's Ratings and an optional query embedding.

Pure ranking: reads Ratings and Titles from SQLite, never calls an LLM or TMDB.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.config import settings
from app.database import get_db, parse_utc_timestamp
from app.services.embeddings import embedding_service
from app.services.tmdb import GENRE_MAP, GENRE_NAME_TO_ID

# Adaptive taste drift (per Spec 0001)
HALF_LIFE_DAYS = 180.0
RECENCY_FLOOR = 0.25
RECENT_WINDOW_SIZE = 10
LONG_TERM_WEIGHT = 0.7
RECENT_WEIGHT = 0.3

# Retrieval tuning balances rating evidence and catalog diversity.
SAME_MEDIA_SUFFICIENT = 3
CROSS_MEDIA_MAX_WEIGHT = 0.25
SHORTLIST_LIMIT = 15
ANCHOR_LIMIT = 6
ANCHOR_NEIGHBORS = 3
AGGREGATE_BUDGET = 30
DIVERSITY_WEIGHT = 0.30
QUERY_DIVERSITY_WEIGHT = 0.05
REDUNDANCY_POWER = 4
DISCOVERY_MIN_CANDIDATES = 8
LOCAL_RELEVANCE_FLOOR = 0.15
ANCHOR_DISTINCTNESS = 0.8
ANCHOR_WEIGHT = 0.65
AGGREGATE_WEIGHT = 0.35
CONTENT_WEIGHT = 0.75
GENRE_WEIGHT = 0.20
DISLIKE_MAX_PENALTY = 0.15
QUALITY_WEIGHT = 0.05
POPULARITY_WEIGHT = 0.03
WATCHLIST_WEIGHT = 0.08
QUERY_WEIGHT = 0.8
RATING_WEIGHTS = {6: 2.0, 5: 1.5, 4: 1.0, 3: -0.2, 2: -1.0, 1: -1.5}

# The LLM understands natural-language categories that TMDB does not expose as
# genres. Translate common user/model vocabulary before applying exact filters.
GENRE_ALIASES: Dict[str, List[str]] = {
    "sitcom": ["Comedy"],
    "romcom": ["Romance", "Comedy"],
    "rom-com": ["Romance", "Comedy"],
    "romantic comedy": ["Romance", "Comedy"],
}

# TMDB merges these movie genres into combined TV genres.
TV_GENRES = {"Science Fiction": "Sci-Fi & Fantasy", "Fantasy": "Sci-Fi & Fantasy",
             "Action": "Action & Adventure", "Adventure": "Action & Adventure"}


@dataclass
class TasteSignal:
    liked_genres: Dict[str, int]
    liked_creators: Dict[str, int]
    liked_count: int
    loved_titles: List[Any]
    rating_rows: List[Any]
    ratings_count: int
    last_rated_at: Optional[str]
    ratings_revision: int


def build_taste_signal() -> TasteSignal:
    """Read all Ratings (newest first) with like statistics."""
    with get_db() as conn:
        conn.execute("BEGIN")
        ratings_revision = conn.execute("SELECT revision FROM ratings_state WHERE id = 1").fetchone()[0]
        rating_rows = conn.execute("""
            SELECT r.id as rating_id, r.score, r.aspect_tags, r.notes,
                   r.created_at, r.updated_at,
                   t.id, t.title, t.media_type, t.release_year, t.genres,
                   t.director_or_creator, t.embedding, t.embedding_model, t.vote_average, t.imdb_rating
            FROM ratings r
            JOIN titles t ON r.title_id = t.id
            ORDER BY COALESCE(r.updated_at, r.created_at) DESC
        """).fetchall()

    liked_genres: Dict[str, int] = {}
    liked_creators: Dict[str, int] = {}
    liked = [r for r in rating_rows if int(r["score"] or 0) >= 4]
    for r in liked:
        for g in json.loads(r["genres"] or "[]"):
            liked_genres[g] = liked_genres.get(g, 0) + 1
        if r["director_or_creator"]:
            liked_creators[r["director_or_creator"]] = liked_creators.get(r["director_or_creator"], 0) + 1
    timestamps = [str(r["updated_at"] or r["created_at"]) for r in rating_rows if r["updated_at"] or r["created_at"]]
    return TasteSignal(
        liked_genres=liked_genres,
        liked_creators=liked_creators,
        liked_count=len(liked),
        loved_titles=[r for r in rating_rows if int(r["score"] or 0) >= 5],
        rating_rows=rating_rows,
        ratings_count=len(rating_rows),
        last_rated_at=max(timestamps, default=None),
        ratings_revision=ratings_revision,
    )


def _recency_decay(timestamp, now: datetime) -> float:
    """Exponential decay with a HALF_LIFE_DAYS half-life, floored at RECENCY_FLOOR."""
    date = parse_utc_timestamp(timestamp)
    if date is None:
        return RECENCY_FLOOR
    age_days = max(0.0, (now - date).total_seconds() / 86400)
    return max(RECENCY_FLOOR, 0.5 ** (age_days / HALF_LIFE_DAYS))


def _unit(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vector) if vector is not None else 0
    return vector / norm if norm else None


def compatible_vector(row: Any) -> Optional[np.ndarray]:
    if not row["embedding"] or row["embedding_model"] != embedding_service.model_key:
        return None
    try:
        vec = embedding_service.bytes_to_vec(row["embedding"])
        if vec.shape == (settings.EMBEDDING_DIM,) and np.isfinite(vec).all():
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else None
    except ValueError:
        pass
    return None


def media_evidence(ratings: List[Any], media: str) -> List[Dict[str, Any]]:
    same = [r for r in ratings if r["media_type"] == media]
    other = [r for r in ratings if r["media_type"] != media]
    cross_weight = CROSS_MEDIA_MAX_WEIGHT * max(0, 1 - len(same) / SAME_MEDIA_SUFFICIENT)
    evidence = []
    now = datetime.now(timezone.utc)
    for rows, media_weight in ((same, 1.0), (other, cross_weight)):
        if not media_weight:
            continue
        group: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            decay = _recency_decay(row["updated_at"] or row["created_at"], now)
            weight = RATING_WEIGHTS.get(row["score"], 0) * (
                LONG_TERM_WEIGHT * decay + RECENT_WEIGHT * (index < RECENT_WINDOW_SIZE))
            group.append({"row": row, "vector": compatible_vector(row), "weight": weight})
        # Cross-media history has a bounded total contribution, regardless of size.
        scale = media_weight / max(1, sum(abs(e["weight"]) for e in group))
        for entry in group:
            entry["weight"] *= scale
        evidence.extend(group)
    return evidence


def positive_anchors(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    remaining = [e for e in entries if e["weight"] > 0 and e["vector"] is not None]
    anchors: List[Dict[str, Any]] = []
    while remaining and len(anchors) < ANCHOR_LIMIT:
        def priority(entry):
            redundancy = max((max(0.0, float(np.dot(entry["vector"], a["vector"])))
                              for a in anchors), default=0.0)
            return entry["weight"] * (1 - ANCHOR_DISTINCTNESS * redundancy), -entry["row"]["id"]
        best = max(remaining, key=priority)
        remaining = [entry for entry in remaining if entry is not best]
        anchors.append(best)
    return anchors


def rank_candidates(intent: Dict[str, Any], signal: Optional[TasteSignal] = None,
                    query_vec: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Return up to SHORTLIST_LIMIT unrated Titles matching `intent`, best first.

    Scores blend query similarity (when given) with media-specific taste: closeness
    to the aggregate and to distinct positive anchors, genre overlap with liked
    Titles, and a conservative penalty for resembling dislikes. The shortlist is
    then picked greedily for diversity (MMR).
    """
    signal = signal or build_taste_signal()
    has_query = query_vec is not None and bool(np.any(query_vec))
    with get_db() as conn:
        rows = conn.execute("""
            SELECT t.*, CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END AS is_on_watchlist
            FROM titles t LEFT JOIN ratings r ON t.id = r.title_id
            LEFT JOIN watchlist w ON t.id = w.title_id
            WHERE r.id IS NULL ORDER BY t.id
        """).fetchall()
    excluded = intent.get("_excluded_title_ids", set())
    rows = [r for r in rows if r["id"] not in excluded and matches_intent(dict(r), intent)]
    vectors = {r["id"]: compatible_vector(r) for r in rows}
    preferred = {g.casefold() for g in intent.get("similarity_genres", [])}

    candidates = []
    neighbors: Dict[tuple, List[tuple]] = {}
    for media in ("movie", "tv"):
        media_rows = [r for r in rows if r["media_type"] == media]
        if not media_rows:
            continue
        evidence = media_evidence(signal.rating_rows, media)
        positive = [e for e in evidence if e["weight"] > 0]
        negative = [e for e in evidence if e["weight"] < 0 and e["vector"] is not None]
        anchors = positive_anchors(evidence)
        positive_weight = sum(e["weight"] for e in positive)
        evidence_strength = min(1.0, positive_weight)
        aggregate = _unit(np.sum([e["vector"] * e["weight"] for e in positive if e["vector"] is not None], axis=0)
                          if any(e["vector"] is not None for e in positive) else None)
        max_anchor_weight = max((a["weight"] for a in anchors), default=1)
        # Liked Titles share few distinct genre sets; score each set once.
        genre_sets: Dict[frozenset, float] = {}
        for e in positive:
            key = frozenset(g.casefold() for g in json.loads(e["row"]["genres"] or "[]"))
            genre_sets[key] = genre_sets.get(key, 0.0) + e["weight"]

        # One matrix product per signal instead of a dot product per (Title, rating) pair.
        embedded = [r for r in media_rows if vectors[r["id"]] is not None]
        index = {r["id"]: i for i, r in enumerate(embedded)}
        matrix = np.vstack([vectors[r["id"]] for r in embedded]) if embedded else np.zeros((0, settings.EMBEDDING_DIM))
        content_dots = matrix @ aggregate if aggregate is not None else np.zeros(len(embedded))
        anchor_affinity = (np.maximum(0.0, matrix @ np.vstack([a["vector"] for a in anchors]).T)
                           * np.array([a["weight"] for a in anchors])) if anchors else np.zeros((len(embedded), 0))
        dislikes = ((np.maximum(0.0, matrix @ np.vstack([e["vector"] for e in negative]).T)
                     * np.array([abs(e["weight"]) for e in negative])).max(axis=1)) if negative else np.zeros(len(embedded))
        query_dots = matrix @ query_vec if query_vec is not None else None

        for row in media_rows:
            i = index.get(row["id"])
            content = anchor_score = dislike = 0.0
            if i is not None:
                content = float(content_dots[i])
                anchor_score = float(anchor_affinity[i].max(initial=0.0)) / max_anchor_weight
                dislike = float(dislikes[i])
                for anchor, affinity in zip(anchors, anchor_affinity[i]):
                    if affinity > 0:
                        neighbors.setdefault((media, anchor["row"]["id"]), []).append((float(affinity), row["id"]))
            content = AGGREGATE_WEIGHT * content + ANCHOR_WEIGHT * anchor_score
            genres = {g.casefold() for g in json.loads(row["genres"] or "[]")}
            genre_affinity = sum(weight * len(genres & rated) / len(genres | rated)
                                 for rated, weight in genre_sets.items() if genres | rated) / max(positive_weight, 1e-9)
            # Whole-Title similarity cannot isolate a complaint about an ending.
            # Keep dislikes conservative and leave qualitative interpretation to the LLM.
            taste_score = (evidence_strength * (CONTENT_WEIGHT * content + GENRE_WEIGHT * genre_affinity)
                           - DISLIKE_MAX_PENALTY * min(1.0, dislike))
            score = taste_score
            if query_dots is not None and i is not None:
                score = QUERY_WEIGHT * float(query_dots[i]) + (1 - QUERY_WEIGHT) * taste_score
            if preferred:
                score += GENRE_WEIGHT * len(genres & preferred) / len(preferred)
            quality = (min((row["vote_average"] or 0) / 10, 1) * QUALITY_WEIGHT
                       + min((row["popularity"] or 0) / 200, 1) * POPULARITY_WEIGHT)
            candidates.append({
                "title_id": row["id"], "tmdb_id": row["tmdb_id"], "media_type": row["media_type"],
                "title": row["title"], "release_year": row["release_year"], "overview": row["overview"],
                "poster_path": row["poster_path"], "genres": json.loads(row["genres"] or "[]"),
                "director": row["director_or_creator"], "cast": json.loads(row["cast_top"] or "[]"),
                "vote_average": row["vote_average"], "vote_count": row["vote_count"], "popularity": row["popularity"],
                "imdb_id": row["imdb_id"], "imdb_rating": row["imdb_rating"] or row["vote_average"] or 0.0,
                "is_on_watchlist": bool(row["is_on_watchlist"]),
                "similarity": score + quality + WATCHLIST_WEIGHT * bool(row["is_on_watchlist"]),
                "raw_similarity": (max(0.0, float(query_dots[i])) if has_query and i is not None
                                   else evidence_strength * max(content, genre_affinity)),
            })

    # With a query, Titles lacking a compatible vector are only kept when none have one.
    if has_query and any(v is not None for v in vectors.values()):
        candidates = [c for c in candidates if vectors[c["title_id"]] is not None]
    ranked = sorted(candidates, key=lambda c: (-c["similarity"], c["title_id"]))
    by_id = {c["title_id"]: c for c in ranked}
    pool_ids: Set[int] = set()
    for media in ("movie", "tv"):
        pool_ids.update(c["title_id"] for c in [c for c in ranked if c["media_type"] == media][:AGGREGATE_BUDGET])
    for group in neighbors.values():
        ordered = sorted((pair for pair in group if pair[1] in by_id),
                         key=lambda pair: (-pair[0], -by_id[pair[1]]["similarity"], pair[1]))
        pool_ids.update(tid for _, tid in ordered[:ANCHOR_NEIGHBORS])
    pool = [c for c in ranked if c["title_id"] in pool_ids]

    selected: List[Dict[str, Any]] = []
    diversity_weight = QUERY_DIVERSITY_WEIGHT if has_query else DIVERSITY_WEIGHT
    while pool and len(selected) < SHORTLIST_LIMIT:
        def priority(candidate):
            vector = vectors[candidate["title_id"]]
            redundancy = max((max(0.0, float(np.dot(vector, vectors[c["title_id"]]))) ** REDUNDANCY_POWER
                              for c in selected if vectors[c["title_id"]] is not None), default=0.0) if vector is not None else 0.0
            return candidate["similarity"] - diversity_weight * redundancy, -candidate["title_id"]
        best = max(pool, key=priority)
        selected.append(best)
        pool.remove(best)
    return selected


def needs_discovery(candidates: List[Dict[str, Any]], has_positive_evidence: bool = True) -> bool:
    if len(candidates) < DISCOVERY_MIN_CANDIDATES:
        return True
    if not has_positive_evidence:
        return False
    relevance = np.asarray([c.get("raw_similarity", 0.0) for c in candidates])
    if embedding_service.is_local:
        # Lexical overlap on the frozen local evaluation fixture, not confidence.
        return bool(np.count_nonzero(relevance >= LOCAL_RELEVANCE_FLOOR) < DISCOVERY_MIN_CANDIDATES)
    # The frozen benchmark validates only local lexical evidence. Until a
    # cloud-specific evaluation establishes relevance, conservatively allow
    # one discovery round rather than interpreting cloud scores as confidence.
    return True


def normalize_intent(intent: Dict[str, Any], query_text: str = "") -> Dict[str, Any]:
    """Convert model vocabulary into safe, canonical TMDB filters.

    Unknown genre labels remain useful in ``semantic_vibe`` but must not be
    exact filters: an invented label would otherwise eliminate every result.
    """
    normalized = dict(intent)
    media_type = normalized.get("media_type")
    query_words = query_text.casefold()
    raw_genres = normalized.get("genres") or []
    canonical_genres: List[str] = []

    for raw_genre in raw_genres:
        if not isinstance(raw_genre, str):
            continue
        genre_key = raw_genre.strip().casefold()
        genres = GENRE_ALIASES.get(genre_key)
        if genres is None and genre_key in GENRE_NAME_TO_ID:
            genres = [GENRE_MAP[GENRE_NAME_TO_ID[genre_key]]]
        for genre in genres or []:
            if media_type == "tv":
                genre = TV_GENRES.get(genre, genre)
            if genre not in canonical_genres:
                canonical_genres.append(genre)

    if "sitcom" in query_words:
        media_type = media_type or "tv"
        if "Comedy" not in canonical_genres:
            canonical_genres.append("Comedy")

    normalized["media_type"] = media_type
    normalized["genres"] = canonical_genres
    normalized["excluded_genres"] = [
        GENRE_MAP[GENRE_NAME_TO_ID[g.strip().casefold()]]
        for g in normalized.get("excluded_genres", [])
        if isinstance(g, str) and g.strip().casefold() in GENRE_NAME_TO_ID
    ]
    normalized["similarity_genres"] = list(dict.fromkeys(
        genre for raw in normalized.get("similarity_genres", [])
        if isinstance(raw, str)
        for genre in GENRE_ALIASES.get(raw.strip().casefold(),
            [GENRE_MAP[GENRE_NAME_TO_ID[raw.strip().casefold()]]]
            if raw.strip().casefold() in GENRE_NAME_TO_ID else [])
        if genre not in normalized["excluded_genres"]
    ))
    # Keep names out of embedding input even if the parser repeats a reference.
    vibe = normalized.get("semantic_vibe", "")
    for title in normalized.get("reference_titles", []):
        if title.strip():
            pattern = r"(?<!\w)" + r"\s*".join(re.escape(word) for word in title.split()) + r"(?!\w)"
            vibe = re.sub(pattern, " ", vibe, flags=re.IGNORECASE)
    normalized["semantic_vibe"] = " ".join(vibe.split()) or " ".join(normalized["similarity_genres"]) or "similar stories and themes"
    return normalized


def matches_intent(title: Dict[str, Any], intent: Dict[str, Any]) -> bool:
    reference_names = {re.sub(r"\W+", "", name).casefold() for name in intent.get("reference_titles", [])}
    if re.sub(r"\W+", "", title.get("title") or "").casefold() in reference_names:
        return False
    if intent.get("media_type") and title.get("media_type") and title["media_type"] != intent["media_type"]:
        return False
    year = title.get("release_year")
    if intent.get("year_min") and (year is None or year < intent["year_min"]):
        return False
    if intent.get("year_max") and (year is None or year > intent["year_max"]):
        return False
    def genre_name(name):
        canonical = GENRE_MAP.get(GENRE_NAME_TO_ID.get(name.casefold()), name)
        if title.get("media_type") == "tv":
            canonical = TV_GENRES.get(canonical, canonical)
        return canonical.casefold()

    genres = title.get("genres") or []
    if isinstance(genres, str):
        genres = json.loads(genres)
    if not {genre_name(g) for g in intent.get("genres") or []}.issubset({genre_name(g) for g in genres}):
        return False
    if {genre_name(g) for g in intent.get("excluded_genres") or []} & {genre_name(g) for g in genres}:
        return False
    person = (intent.get("person") or "").strip().casefold()
    if person:
        cast = title.get("cast_top") or []
        if isinstance(cast, str):
            cast = json.loads(cast)
        people = [title.get("director_or_creator") or "", *cast]
        if not any(person == name.strip().casefold() for name in people):
            return False
    return True

