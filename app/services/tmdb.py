import logging
from typing import List, Dict, Any, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

GENRE_MAP = {
    # Movie & TV TMDB Genre IDs
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
    10759: "Action & Adventure",
    10762: "Kids",
    10763: "News",
    10764: "Reality",
    10765: "Sci-Fi & Fantasy",
    10766: "Soap",
    10767: "Talk",
    10768: "War & Politics",
}

GENRE_NAME_TO_ID = {v.lower(): k for k, v in GENRE_MAP.items()}
GENRE_NAME_TO_ID["sci-fi"] = 878


def _year(date_str: Optional[str]) -> Optional[int]:
    return int(date_str[:4]) if date_str and date_str[:4].isdigit() else None


def _parse_item(item: Dict[str, Any], media_type: str) -> Dict[str, Any]:
    """Normalize a TMDB list/search result into the app's title shape."""
    is_movie = media_type == "movie"
    return {
        "tmdb_id": item["id"],
        "media_type": media_type,
        "title": (item.get("title") if is_movie else item.get("name")) or "Untitled",
        "original_title": item.get("original_title") if is_movie else item.get("original_name"),
        "release_year": _year(item.get("release_date") if is_movie else item.get("first_air_date")),
        "overview": item.get("overview") or "",
        "poster_path": item.get("poster_path"),
        "backdrop_path": item.get("backdrop_path"),
        "genres": [GENRE_MAP[gid] for gid in item.get("genre_ids", []) if gid in GENRE_MAP],
        "vote_average": float(item.get("vote_average", 0.0)),
        "vote_count": int(item.get("vote_count", 0)),
        "popularity": float(item.get("popularity", 0.0)),
    }


class TMDBService:
    def __init__(self):
        self.api_key = settings.TMDB_API_KEY
        self.base_url = settings.TMDB_BASE_URL
        self.image_base_url = settings.TMDB_IMAGE_BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        # v4 read tokens are long bearer tokens; v3 keys go in the query string.
        if len(self.api_key) > 50:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = {"api_key": self.api_key} if 0 < len(self.api_key) <= 50 else {}
        return {**params, **(extra or {})}

    async def _get(self, client: httpx.AsyncClient, path: str, **params) -> Optional[Dict[str, Any]]:
        """GET a TMDB endpoint; log and return None on any failure."""
        try:
            response = await client.get(f"{self.base_url}/{path}", params=self._get_params(params),
                                        headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"TMDB request {path} failed: {type(e).__name__}")
            return None

    async def _get_once(self, path: str, timeout: float = 10.0, **params) -> Optional[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await self._get(client, path, **params)

    def get_poster_url(self, poster_path: Optional[str]) -> str:
        if not poster_path:
            return "/static/images/placeholder_poster.svg"
        if poster_path.startswith("http"):
            return poster_path
        return f"{self.image_base_url}{poster_path}"

    async def search_titles(self, query: str, page: int = 1) -> List[Dict[str, Any]]:
        """Search movies and TV series across TMDB."""
        if not query.strip() or not self.api_key:
            return []
        data = await self._get_once("search/multi", query=query.strip(), page=page, include_adult=False) or {}
        return [_parse_item(item, item["media_type"]) for item in data.get("results", [])
                if item.get("media_type") in ("movie", "tv")]

    async def get_title_details(self, tmdb_id: int, media_type: str) -> Optional[Dict[str, Any]]:
        """Fetch full title details with credits and directors."""
        if not self.api_key:
            return None
        data = await self._get_once(f"{media_type}/{tmdb_id}", timeout=12.0,
                                    append_to_response="credits,external_ids")
        if not data:
            return None

        crew = data.get("credits", {}).get("crew", [])
        if media_type == "movie":
            director_or_creator = next((c.get("name") for c in crew if c.get("job") == "Director"), None)
        else:
            creators = [c.get("name") for c in data.get("created_by", []) if c.get("name")]
            director_or_creator = ", ".join(creators) or next(
                (c.get("name") for c in crew if c.get("job") in ("Executive Producer", "Series Director")), None)

        title = _parse_item(data, media_type)
        title.update({
            "genres": [g["name"] for g in data.get("genres", []) if "name" in g],
            "director_or_creator": director_or_creator,
            "cast_top": [c.get("name") for c in data.get("credits", {}).get("cast", [])[:5] if c.get("name")],
            "imdb_id": data.get("imdb_id") or data.get("external_ids", {}).get("imdb_id"),
            "imdb_rating": title["vote_average"],
        })
        return title

    async def get_famous_titles(
        self,
        min_vote_count: int = 1000,
        min_vote_average: float = 6.5,
        pages_per_endpoint: int = 2,
        limit: int = 40,
        start_page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Fetch widely recognized, well-rated Titles for the Rating Deck.

        Walks top_rated + popular for movies and TV starting at `start_page`, so
        successive Deck refills reach deeper into the catalog instead of
        re-fetching the same first pages.
        """
        if not self.api_key:
            return []

        seen: set[tuple[int, str]] = set()
        collected: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for media_type in ("movie", "tv"):
                for category in ("top_rated", "popular"):
                    for page in range(start_page, start_page + pages_per_endpoint):
                        data = await self._get(client, f"{media_type}/{category}", page=page, include_adult=False)
                        if data is None:
                            break
                        # Deeper pages relax the vote floor a little, never below 400.
                        min_votes = max(400, min_vote_count - (page // 5) * 50)
                        for item in data.get("results", []):
                            if item.get("id") is None or (item["id"], media_type) in seen:
                                continue
                            seen.add((item["id"], media_type))
                            title = _parse_item(item, media_type)
                            if title["vote_count"] >= min_votes and title["vote_average"] >= min_vote_average:
                                collected.append(title)
                        if page >= int(data.get("total_pages", page)):
                            break

        collected.sort(key=lambda x: (x["vote_count"], x["popularity"]), reverse=True)
        return collected[:limit]

    async def get_title_recommendations(self, tmdb_id: int, media_type: str = "movie",
                                        limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch recommended titles from TMDB based on a specific title."""
        if not self.api_key or not tmdb_id:
            return []
        data = await self._get_once(f"{media_type}/{tmdb_id}/recommendations", page=1) or {}
        return [_parse_item(item, media_type) for item in data.get("results", [])[:limit]]

    async def find_person_titles(self, name: str, media_type: str) -> List[Dict[str, Any]]:
        """Find acting/directing credits before fetching full title details."""
        if not self.api_key:
            return []
        async with httpx.AsyncClient(timeout=10.0) as client:
            people = (await self._get(client, "search/person", query=name) or {}).get("results", [])
            person = next((p for p in people if p["name"].casefold() == name.strip().casefold()), None)
            if person is None:
                return []
            credits = await self._get(client, f"person/{person['id']}/combined_credits") or {}

        results = {}
        for role in ("cast", "crew"):
            for item in credits.get(role, []):
                if item.get("media_type") != media_type or item.get("adult"):
                    continue
                if role == "crew" and item.get("job") not in ("Director", "Creator"):
                    continue
                date = item.get("release_date") or item.get("first_air_date")
                title = results.setdefault(item["id"], {
                    "tmdb_id": item["id"], "media_type": media_type,
                    "title": item.get("title") or item.get("name") or "Untitled",
                    "release_year": _year(date),
                    "genres": [GENRE_MAP[g] for g in item.get("genre_ids", []) if g in GENRE_MAP],
                    "overview": item.get("overview", ""), "poster_path": item.get("poster_path"),
                    "popularity": item.get("popularity", 0), "vote_average": item.get("vote_average", 0),
                    "cast_top": [], "director_or_creator": None,
                })
                if role == "cast":
                    title["cast_top"] = [person["name"]]
                else:
                    title["director_or_creator"] = person["name"]
        return sorted(results.values(), key=lambda t: t["popularity"], reverse=True)

    async def discover_titles(
        self,
        media_type: str = "movie",
        with_genres: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        sort_by: str = "popularity.desc",
        vote_count_gte: int = 200,
        vote_average_gte: Optional[float] = None,
        page: int = 1,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Dynamic discover query for recommendation candidates."""
        if not self.api_key:
            return []

        params: Dict[str, Any] = {"sort_by": sort_by, "page": page,
                                  "vote_count.gte": vote_count_gte, "include_adult": False}
        if with_genres:
            params["with_genres"] = with_genres
        if vote_average_gte:
            params["vote_average.gte"] = vote_average_gte
        date_field = "primary_release_date" if media_type == "movie" else "first_air_date"
        if year_min:
            params[f"{date_field}.gte"] = f"{year_min}-01-01"
        if year_max:
            params[f"{date_field}.lte"] = f"{year_max}-12-31"

        data = await self._get_once(f"discover/{media_type}", **params) or {}
        return [_parse_item(item, media_type) for item in data.get("results", [])[:limit]]


tmdb_service = TMDBService()
