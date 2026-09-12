"""Load optional distribution defaults without storing credentials in Git."""

import json
from importlib.resources import files

# Release builds include this Git-ignored resource. Source checkouts use .env.
_resource = files("app").joinpath("data/tmdb_default.json")
DEFAULT_TMDB_API_KEY = json.loads(_resource.read_text(encoding="utf-8"))["api_key"] if _resource.is_file() else ""
