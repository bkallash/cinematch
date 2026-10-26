from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class RatingCreate(BaseModel):
    title_id: int
    score: int = Field(ge=1, le=6, description="Rating from 1 to 6")
    aspect_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class QueryIntent(BaseModel):
    model_config = ConfigDict(strict=True)

    semantic_vibe: str
    reference_titles: List[str] = Field(default_factory=list)
    similarity_genres: List[str] = Field(default_factory=list)
    media_type: Optional[Literal["movie", "tv"]] = None
    genres: List[str] = Field(default_factory=list)
    person: Optional[str] = None
    excluded_genres: List[str] = Field(default_factory=list)
    year_min: Optional[int] = Field(default=None, ge=1800, le=2200)
    year_max: Optional[int] = Field(default=None, ge=1800, le=2200)


class ModelPick(BaseModel):
    model_config = ConfigDict(strict=True)

    title_id: int
    title: str = Field(min_length=1)
    reason: str = ""


class DossierContent(BaseModel):
    model_config = ConfigDict(strict=True)

    core_loves: List[str] = Field(default_factory=list)
    deal_breakers: List[str] = Field(default_factory=list)
    creator_affinities: List[str] = Field(default_factory=list)
    atmospheric_preferences: List[str] = Field(default_factory=list)
    narrative_tropes: List[str] = Field(default_factory=list)
    full_summary: str = Field(min_length=1)
