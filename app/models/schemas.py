from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class TitleBase(BaseModel):
    tmdb_id: int
    media_type: Literal["movie", "tv"]
    title: str
    original_title: Optional[str] = None
    release_year: Optional[int] = None
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    director_or_creator: Optional[str] = None
    cast_top: List[str] = Field(default_factory=list)
    vote_average: float = 0.0
    vote_count: int = 0
    popularity: float = 0.0
    imdb_id: Optional[str] = None
    imdb_rating: Optional[float] = None

class TitleCreate(TitleBase):
    pass

class TitleRead(TitleBase):
    id: int
    created_at: Optional[datetime] = None
    user_rating: Optional[int] = None
    user_aspect_tags: List[str] = Field(default_factory=list)
    user_notes: Optional[str] = None
    is_in_watchlist: bool = False

class RatingCreate(BaseModel):
    title_id: int
    score: int = Field(ge=1, le=6, description="Rating from 1 to 6")
    aspect_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class RatingRead(BaseModel):
    id: int
    title_id: int
    score: int
    aspect_tags: List[str]
    notes: Optional[str] = None
    created_at: datetime
    title: Optional[TitleRead] = None

class WatchlistCreate(BaseModel):
    title_id: int
    notes: Optional[str] = None

class WatchlistRead(BaseModel):
    id: int
    title_id: int
    notes: Optional[str] = None
    created_at: datetime
    title: Optional[TitleRead] = None

class TasteDossierRead(BaseModel):
    user_id: str
    core_loves: List[str]
    deal_breakers: List[str]
    creator_affinities: List[str]
    atmospheric_preferences: List[str]
    narrative_tropes: List[str]
    full_summary: str
    ratings_count_at_synthesis: int
    is_dirty: bool
    updated_at: datetime

class VibeQueryRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    media_type: Literal["all", "movie", "tv"] = "all"

class RecommendedTitle(BaseModel):
    title_id: int
    tmdb_id: int
    media_type: str
    title: str
    release_year: Optional[int] = None
    poster_path: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    overview: Optional[str] = None
    reason: str
    is_on_watchlist: bool = False
    vote_average: Optional[float] = None
    imdb_rating: Optional[float] = None
    imdb_id: Optional[str] = None

class RecommendationResponse(BaseModel):
    assistant_message: str
    recommendations: List[RecommendedTitle]
    session_id: str


class QueryIntent(BaseModel):
    model_config = ConfigDict(strict=True)

    semantic_vibe: str
    media_type: Optional[Literal["movie", "tv"]] = None
    genres: List[str] = Field(default_factory=list)
    person: Optional[str] = None
    year_min: Optional[int] = Field(default=None, ge=1800, le=2200)
    year_max: Optional[int] = Field(default=None, ge=1800, le=2200)


class ModelPick(BaseModel):
    model_config = ConfigDict(strict=True)

    title_id: int
    reason: str = ""


class DossierContent(BaseModel):
    model_config = ConfigDict(strict=True)

    core_loves: List[str] = Field(default_factory=list)
    deal_breakers: List[str] = Field(default_factory=list)
    creator_affinities: List[str] = Field(default_factory=list)
    atmospheric_preferences: List[str] = Field(default_factory=list)
    narrative_tropes: List[str] = Field(default_factory=list)
    full_summary: str = Field(min_length=1)
