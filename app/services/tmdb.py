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
GENRE_NAME_TO_ID["science fiction"] = 878
GENRE_NAME_TO_ID["sci-fi & fantasy"] = 10765
GENRE_NAME_TO_ID["action & adventure"] = 10759

class TMDBService:
    def __init__(self):
        self.api_key = settings.TMDB_API_KEY
        self.base_url = settings.TMDB_BASE_URL
        self.image_base_url = settings.TMDB_IMAGE_BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        # Check if API key is a v4 Bearer token (usually ~150-200 chars) or standard v3 key
        if len(self.api_key) > 50:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = {}
        if len(self.api_key) <= 50 and self.api_key:
            params["api_key"] = self.api_key
        if extra:
            params.update(extra)
        return params

    def get_poster_url(self, poster_path: Optional[str]) -> str:
        if not poster_path:
            return "/static/images/placeholder_poster.svg"
        if poster_path.startswith("http"):
            return poster_path
        return f"{self.image_base_url}{poster_path}"

    async def search_titles(self, query: str, page: int = 1) -> List[Dict[str, Any]]:
        """Search movies and TV series across TMDB."""
        if not query or not query.strip():
            return []
        if not self.api_key:
            logger.warning("TMDB_API_KEY is not configured.")
            return []

        url = f"{self.base_url}/search/multi"
        params = self._get_params({"query": query.strip(), "page": page, "include_adult": False})

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Error searching TMDB: {e}")
                return []

        results = []
        for item in data.get("results", []):
            media_type = item.get("media_type")
            if media_type not in ("movie", "tv"):
                continue

            title_name = item.get("title") if media_type == "movie" else item.get("name")
            orig_name = item.get("original_title") if media_type == "movie" else item.get("original_name")
            date_str = item.get("release_date") if media_type == "movie" else item.get("first_air_date")

            release_year = None
            if date_str and len(date_str) >= 4:
                try:
                    release_year = int(date_str[:4])
                except ValueError:
                    pass

            genre_ids = item.get("genre_ids", [])
            genres = [GENRE_MAP[gid] for gid in genre_ids if gid in GENRE_MAP]

            results.append({
                "tmdb_id": item["id"],
                "media_type": media_type,
                "title": title_name or "Untitled",
                "original_title": orig_name,
                "release_year": release_year,
                "overview": item.get("overview") or "",
                "poster_path": item.get("poster_path"),
                "backdrop_path": item.get("backdrop_path"),
                "genres": genres,
                "vote_average": float(item.get("vote_average", 0.0)),
                "vote_count": int(item.get("vote_count", 0)),
                "popularity": float(item.get("popularity", 0.0)),
            })
        return results

    async def get_title_details(self, tmdb_id: int, media_type: str) -> Optional[Dict[str, Any]]:
        """Fetch full title details with credits and directors."""
        if not self.api_key:
            return None

        endpoint = "movie" if media_type == "movie" else "tv"
        url = f"{self.base_url}/{endpoint}/{tmdb_id}"
        params = self._get_params({"append_to_response": "credits,external_ids"})

        async with httpx.AsyncClient(timeout=12.0) as client:
            try:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Error fetching TMDB details for {media_type}:{tmdb_id}: {e}")
                return None

        title_name = data.get("title") if media_type == "movie" else data.get("name")
        orig_name = data.get("original_title") if media_type == "movie" else data.get("original_name")
        date_str = data.get("release_date") if media_type == "movie" else data.get("first_air_date")

        release_year = None
        if date_str and len(date_str) >= 4:
            try:
                release_year = int(date_str[:4])
            except ValueError:
                pass

        genres = [g["name"] for g in data.get("genres", []) if "name" in g]

        credits_data = data.get("credits", {})
        director_or_creator = None

        if media_type == "movie":
            for crew_member in credits_data.get("crew", []):
                if crew_member.get("job") == "Director":
                    director_or_creator = crew_member.get("name")
                    break
        else:
            creators = [c.get("name") for c in data.get("created_by", []) if c.get("name")]
            if creators:
                director_or_creator = ", ".join(creators)
            else:
                for crew_member in credits_data.get("crew", []):
                    if crew_member.get("job") in ("Executive Producer", "Series Director"):
                        director_or_creator = crew_member.get("name")
                        break

        cast_top = [c.get("name") for c in credits_data.get("cast", [])[:5] if c.get("name")]
        imdb_id = data.get("imdb_id") or data.get("external_ids", {}).get("imdb_id")

        return {
            "tmdb_id": data["id"],
            "media_type": media_type,
            "title": title_name or "Untitled",
            "original_title": orig_name,
            "release_year": release_year,
            "overview": data.get("overview") or "",
            "poster_path": data.get("poster_path"),
            "backdrop_path": data.get("backdrop_path"),
            "genres": genres,
            "director_or_creator": director_or_creator,
            "cast_top": cast_top,
            "vote_average": float(data.get("vote_average", 0.0)),
            "vote_count": int(data.get("vote_count", 0)),
            "popularity": float(data.get("popularity", 0.0)),
            "imdb_id": imdb_id,
            "imdb_rating": float(data.get("vote_average", 0.0)),
        }

    async def get_famous_titles(
        self,
        min_vote_count: int = 1000,
        min_vote_average: float = 6.5,
        pages_per_endpoint: int = 2,
        limit: int = 40,
        start_page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Fetch universally recognized, well-rated Titles for the Rating Deck.

        Hits TMDB top_rated + popular endpoints for movies and TV, filters out
        obscure/low-rated long-tail results (which otherwise pollute the Deck
        after a free-text search caches every lookalike), dedupes, and returns
        the most-voted candidates first.

        `start_page` sets the first TMDB page fetched per endpoint so Deck
        refills can walk deeper into the catalog on every call instead of
        re-fetching pages 1-2 (which would stop yielding fresh Titles and let
        the Deck complete).
        """
        if not self.api_key:
            return []

        endpoints = [
            ("movie", "top_rated"),
            ("movie", "popular"),
            ("tv", "top_rated"),
            ("tv", "popular"),
        ]

        seen: set[tuple[int, str]] = set()
        collected: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for media_type, category in endpoints:
                endpoint = "movie" if media_type == "movie" else "tv"
                for page in range(start_page, start_page + pages_per_endpoint):
                    url = f"{self.base_url}/{endpoint}/{category}"
                    params = self._get_params({
                        "page": page,
                        "include_adult": False,
                    })
                    try:
                        response = await client.get(url, params=params, headers=self._get_headers())
                        response.raise_for_status()
                        data = response.json()
                    except Exception as e:
                        logger.error(f"Error fetching TMDB {endpoint}/{category} p{page}: {e}")
                        break

                    for item in data.get("results", []):
                        tmdb_id = item.get("id")
                        if tmdb_id is None:
                            continue
                        key = (tmdb_id, media_type)
                        if key in seen:
                            continue
                        seen.add(key)

                        vote_average = float(item.get("vote_average", 0.0))
                        vote_count = int(item.get("vote_count", 0))
                        effective_min_vote_count = max(400, min_vote_count - (page // 5) * 50)
                        if vote_count < effective_min_vote_count or vote_average < min_vote_average:
                            continue

                        title_name = item.get("title") if media_type == "movie" else item.get("name")
                        date_str = item.get("release_date") if media_type == "movie" else item.get("first_air_date")
                        release_year = None
                        if date_str and len(date_str) >= 4:
                            try:
                                release_year = int(date_str[:4])
                            except ValueError:
                                pass

                        genre_ids = item.get("genre_ids", [])
                        genres = [GENRE_MAP[gid] for gid in genre_ids if gid in GENRE_MAP]

                        collected.append({
                            "tmdb_id": tmdb_id,
                            "media_type": media_type,
                            "title": title_name or "Untitled",
                            "original_title": item.get("original_title") or item.get("original_name"),
                            "release_year": release_year,
                            "overview": item.get("overview") or "",
                            "poster_path": item.get("poster_path"),
                            "backdrop_path": item.get("backdrop_path"),
                            "genres": genres,
                            "vote_average": vote_average,
                            "vote_count": vote_count,
                            "popularity": float(item.get("popularity", 0.0)),
                        })

                    # Stop paging this endpoint when TMDB reports last page
                    if page >= int(data.get("total_pages", page)):
                        break

        collected.sort(key=lambda x: (x["vote_count"], x["popularity"]), reverse=True)
        return collected[:limit]

    async def get_title_recommendations(
        self,
        tmdb_id: int,
        media_type: str = "movie",
        limit: int = 15
    ) -> List[Dict[str, Any]]:
        """Fetch recommended titles from TMDB based on a specific title."""
        if not self.api_key or not tmdb_id:
            return []

        endpoint = "movie" if media_type == "movie" else "tv"
        url = f"{self.base_url}/{endpoint}/{tmdb_id}/recommendations"
        params = self._get_params({"page": 1})

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Error fetching TMDB recommendations for {media_type}:{tmdb_id}: {e}")
                return []

        results = []
        for item in data.get("results", [])[:limit]:
            title_name = item.get("title") if media_type == "movie" else item.get("name")
            date_str = item.get("release_date") if media_type == "movie" else item.get("first_air_date")
            release_year = None
            if date_str and len(date_str) >= 4:
                try:
                    release_year = int(date_str[:4])
                except ValueError:
                    pass

            genre_ids = item.get("genre_ids", [])
            genres = [GENRE_MAP[gid] for gid in genre_ids if gid in GENRE_MAP]

            results.append({
                "tmdb_id": item["id"],
                "media_type": media_type,
                "title": title_name or "Untitled",
                "original_title": item.get("original_title") or item.get("original_name"),
                "release_year": release_year,
                "overview": item.get("overview") or "",
                "poster_path": item.get("poster_path"),
                "backdrop_path": item.get("backdrop_path"),
                "genres": genres,
                "vote_average": float(item.get("vote_average", 0.0)),
                "vote_count": int(item.get("vote_count", 0)),
                "popularity": float(item.get("popularity", 0.0)),
            })
        return results

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

        endpoint = "movie" if media_type == "movie" else "tv"
        url = f"{self.base_url}/discover/{endpoint}"
        extra_params: Dict[str, Any] = {
            "sort_by": sort_by,
            "page": page,
            "vote_count.gte": vote_count_gte,
            "include_adult": False
        }

        if with_genres:
            extra_params["with_genres"] = with_genres
        if vote_average_gte:
            extra_params["vote_average.gte"] = vote_average_gte
        if year_min:
            date_field = "primary_release_date.gte" if media_type == "movie" else "first_air_date.gte"
            extra_params[date_field] = f"{year_min}-01-01"
        if year_max:
            date_field = "primary_release_date.lte" if media_type == "movie" else "first_air_date.lte"
            extra_params[date_field] = f"{year_max}-12-31"

        params = self._get_params(extra_params)

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Error discovering TMDB {media_type}: {e}")
                return []

        results = []
        for item in data.get("results", [])[:limit]:
            title_name = item.get("title") if media_type == "movie" else item.get("name")
            date_str = item.get("release_date") if media_type == "movie" else item.get("first_air_date")
            release_year = None
            if date_str and len(date_str) >= 4:
                try:
                    release_year = int(date_str[:4])
                except ValueError:
                    pass

            genre_ids = item.get("genre_ids", [])
            genres = [GENRE_MAP[gid] for gid in genre_ids if gid in GENRE_MAP]

            results.append({
                "tmdb_id": item["id"],
                "media_type": media_type,
                "title": title_name or "Untitled",
                "original_title": item.get("original_title") or item.get("original_name"),
                "release_year": release_year,
                "overview": item.get("overview") or "",
                "poster_path": item.get("poster_path"),
                "backdrop_path": item.get("backdrop_path"),
                "genres": genres,
                "vote_average": float(item.get("vote_average", 0.0)),
                "vote_count": int(item.get("vote_count", 0)),
                "popularity": float(item.get("popularity", 0.0)),
            })
        return results

tmdb_service = TMDBService()
