from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from app.bundled_config import DEFAULT_TMDB_API_KEY
from app.cli import default_data_dir

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # OpenRouter
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API Key")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter Base URL")
    OPENROUTER_MODEL: str = Field(default="openai/gpt-4o-mini", description="Model for reasoning and chat")

    # Embeddings
    EMBEDDING_PROVIDER: Literal["openrouter", "local"] = "openrouter"
    EMBEDDING_MODEL: str = Field(default="openai/text-embedding-3-small", description="Model for text embeddings")
    EMBEDDING_DIM: int = Field(default=1536, gt=0, description="Dimension of embedding vectors")

    # TMDB API
    TMDB_API_KEY: str = Field(default=DEFAULT_TMDB_API_KEY, description="The Movie Database API Key")
    TMDB_BASE_URL: str = Field(default="https://api.themoviedb.org/3", description="TMDB API Base URL")
    TMDB_IMAGE_BASE_URL: str = Field(default="https://image.tmdb.org/t/p/w500", description="TMDB Poster CDN Base URL")

    # Database
    # Same default as the CLI, so running uvicorn directly never writes into the installed package.
    DATABASE_PATH: Path = Field(default_factory=lambda: default_data_dir() / "cinematch.db")

    # App Settings
    APP_PORT: int = Field(default=8000)

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
