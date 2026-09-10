import asyncio

import pytest

from app.config import settings
from app.database import init_db
from app.services.seeder import seed_starter_catalog
from app.services.tmdb import tmdb_service


@pytest.fixture(scope="session", autouse=True)
def offline_test_database(tmp_path_factory):
    """Legacy integration tests share a seeded database, never the user's library."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "DATABASE_PATH", tmp_path_factory.mktemp("cinematch") / "test.db")
        patch.setattr(settings, "OPENROUTER_API_KEY", "")
        patch.setattr(settings, "TMDB_API_KEY", "")
        patch.setattr(settings, "EMBEDDING_PROVIDER", "openrouter")
        patch.setattr(settings, "EMBEDDING_DIM", 1536)
        patch.setattr(tmdb_service, "api_key", "")
        init_db()
        asyncio.run(seed_starter_catalog())
        yield
