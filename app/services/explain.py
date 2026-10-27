"""Explain recommendations: heuristic reasons and validation of LLM picks."""
import json
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.models.schemas import ModelPick

# Phrases that mark an LLM reason as too generic to show; a heuristic reason replaces it.
GENERIC_REASON_MARKERS = ("vector", "matches your taste", "taste profile")


def is_generic_reason(reason: str, extra_markers: tuple = ()) -> bool:
    lowered = reason.lower()
    return len(reason.strip()) < 15 or any(m in lowered for m in GENERIC_REASON_MARKERS + extra_markers)


def rec_card(candidate: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """The subset of a candidate the chat UI renders as a recommendation card."""
    return {
        "title_id": candidate["title_id"], "tmdb_id": candidate["tmdb_id"],
        "media_type": candidate["media_type"], "title": candidate["title"],
        "release_year": candidate["release_year"], "poster_path": candidate["poster_path"],
        "genres": candidate["genres"], "overview": candidate["overview"], "reason": reason,
        "is_on_watchlist": candidate.get("is_on_watchlist", False),
        "vote_average": candidate.get("vote_average", 0.0),
        "imdb_rating": candidate.get("imdb_rating") or candidate.get("vote_average", 0.0),
        "imdb_id": candidate.get("imdb_id"),
    }


def personalized_reason(
    candidate: Dict[str, Any],
    liked_genres: Optional[Dict[str, int]] = None,
    liked_creators: Optional[Dict[str, int]] = None,
    loved_titles: Optional[List[Any]] = None,
) -> str:
    """Constructs a film-literate, compelling explanation of WHY this Title is recommended.

    Prioritizes creator affinity, then genres shared with a loved title, then
    liked genres, then watchlist status. Never exposes technical jargon.
    """
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
            return f"By {c_director}, whose work you have rated positively, and already on your Watchlist."
        return f"By {c_director}, whose work you have rated positively."

    # 2. Connections with specific user loved titles (rated 5 or 6)
    if loved_titles:
        best_overlap_loved = None
        best_overlap_count = 0
        best_shared_genres = []
        for loved in loved_titles:
            l_genres = loved["genres"]
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
            loved_name = best_overlap_loved["title"]
            shared_str = " & ".join(best_shared_genres[:2])
            if is_wl:
                return f"Shares the {shared_str} genres with your highly rated '{loved_name}' and is on your Watchlist."
            return f"Shares the {shared_str} genres with your highly rated '{loved_name}'; it may be worth exploring."
        elif best_overlap_loved and best_overlap_count == 1:
            loved_name = best_overlap_loved["title"]
            shared_genre = best_shared_genres[0]
            if is_wl:
                return f"Like your highly rated '{loved_name}', this is a {shared_genre} Title, already on your Watchlist."
            return f"Shares the {shared_genre} genre with your highly rated '{loved_name}'; other aspects may differ."

    # 3. Genre affinity match
    if liked_genres:
        matching_genres = [g for g in c_genres if g in liked_genres]
        if matching_genres:
            matching_genres.sort(key=lambda g: liked_genres.get(g, 0), reverse=True)
            genre_str = " & ".join(matching_genres[:2])
            if is_wl:
                return f"A {genre_str} Title on your Watchlist; you have rated other Titles in these genres positively."
            if c_score >= 7.5:
                return f"A {genre_str} Title with an audience rating of {c_score:.1f}/10; you have liked other Titles in these genres."
            return f"A {genre_str} Title to explore, based on your positive ratings in these genres."

    # A dossier preference alone is not evidence that this candidate has it.
    if is_wl:
        return "You saved this Title to your Watchlist; it is still waiting to be watched."
    if c_genres:
        return f"A {' / '.join(c_genres[:2])} option to explore. Rate it to help refine future recommendations."
    return "An unrated Title to explore; there is not enough evidence for a more specific connection yet."


def heuristic_reasons(candidates: List[Dict[str, Any]], liked_genres: Dict[str, int],
                      liked_creators: Dict[str, int], loved_titles: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Explain each pick via genre/creator overlap and connection to user favorites."""
    return [{**c, "reason": personalized_reason(c, liked_genres, liked_creators, loved_titles)} for c in candidates]


def candidate_prompt_data(candidates):
    # Named fields prevent confusing list positions or display labels with identity.
    fields = ("title_id", "title", "release_year", "media_type", "genres", "director",
              "cast", "overview", "vote_average", "is_on_watchlist")
    return [{key: candidate.get(key) for key in fields} for candidate in candidates]


def valid_picks(entries, candidates=None):
    if not isinstance(entries, list):
        return []
    valid = []
    for entry in entries:
        try:
            pick = ModelPick.model_validate(entry).model_dump()
            if candidates is not None and pick.get("title") is not None:
                candidate = candidates.get(pick["title_id"])
                if not candidate or pick["title"].strip().casefold() != candidate["title"].strip().casefold():
                    continue
            valid.append(pick)
        except ValidationError:
            continue
    return valid


def accepted_picks(data: Dict[str, Any], key: str, candidates: Dict[int, Any]) -> List[tuple]:
    """(title_id, reason) for valid, unique, non-rejected model picks, in model order."""
    entries = data.get("rejected_title_ids", [])
    rejected = {tid for tid in entries if type(tid) is int} if isinstance(entries, list) else set()
    accepted = {}
    for pick in valid_picks(data.get(key, []), candidates):
        if pick["title_id"] in candidates and pick["title_id"] not in rejected:
            accepted.setdefault(pick["title_id"], str(pick.get("reason", "")).strip())
    return list(accepted.items())

