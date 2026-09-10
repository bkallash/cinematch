# CineMatch — AI-Powered Movie & Series Suggestion App

A modern, cinephile-grade recommendation engine built in Python, featuring a living **Taste Dossier**, hybrid vector RAG retrieval, and an interactive dark-mode web UI.

---

## Key Features

1. **💬 CineMatch Concierge (Assistant)**:
   - Natural language vibe matching (e.g. *"Suggest a romantic movie like La La Land with bittersweet melancholia"*, *"Cozy 90s holiday comedy"*).
   - Multi-turn conversational flow powered by OpenRouter **GPT-4o**.
   - Dual-layer recommendation pipeline: fast in-process SQLite vector retrieval + LLM reranking against your living **Taste Dossier**.
   - Surfaces personalized reasoning cards with "Why You'll Love It" callouts and highlights matching titles from your own Watchlist.

2. **🎴 Famous Titles Rating Deck**:
   - Fast-fire card queue of widely recognized movies and TV series with posters, trailers, and synopses.
   - Non-neutral **1 to 6 rating scale** (1–3 = Disliked/Flawed, 4–6 = Good/Masterpiece).
   - Adaptive **Aspect Tags** (concise single-concept tags like `Visuals`, `Pacing`, `Ending`, `Soundtrack`, `Plot Holes`, `Cliche`).
   - Quick **"Haven't Seen (Skip)"** and **"Want to Watch"** action buttons.

3. **🔍 Search & Rate**:
   - Real-time debounced search bar connecting local SQLite storage with live TMDB (The Movie Database).
   - Log 1–6 ratings and micro-notes on any title in cinema history.

4. **📚 My Library & Taste Dossier**:
   - **Ratings**: Browse and filter your rated titles by score, media type, and aspect tags.
   - **Watchlist**: Manage everything marked "Want to Watch".
   - **Taste Dossier**: Transparently inspect what the AI has synthesized about your cinematic DNA (Core Loves, Deal-Breakers, Creator Affinities, and Narrative Tropes), with a one-click re-synthesis trigger.

---

## Quickstart Guide

### 1. Activate the Virtual Environment

```powershell
cd "D:\code Projects\SPOC\movie-suggestion"
.\venv\Scripts\activate
```

### 2. Configure Your API Keys in `.env`

Copy `.env.example` to `.env` (or edit existing `.env`):

```env
# OpenRouter Configuration (for GPT-4o and Text Embeddings)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o

# TMDB API Key (Free from https://www.themoviedb.org/settings/api)
TMDB_API_KEY=your_tmdb_api_key_here

# Embeddings
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=openai/text-embedding-3-small
```

> **Note**: Even before adding API keys, CineMatch runs fully with its pre-bundled catalog of famous titles and deterministic vector fallbacks.

Set `EMBEDDING_PROVIDER=local` for offline token-based text matching without a model download. Cloud embeddings provide richer semantic matching. Each stored vector tracks its provider, model, and dimension; startup rebuilds incompatible or older unversioned vectors in batches of 64. With OpenRouter configured, this rebuild makes billable embedding requests. Failed batches are retried at the next startup.

### 3. Run the Web Application

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Open your browser at: **[http://localhost:8000](http://localhost:8000)**

---

## Install from PyPI

Install and run CineMatch from any directory:

```powershell
python -m pip install cinematch
cinematch
```

On the first run, CineMatch prompts for:

1. Your OpenRouter API key. Create one at [OpenRouter Keys](https://openrouter.ai/settings/keys). The key is entered using a hidden password prompt.
2. An OpenRouter model identifier, such as `openai/gpt-4o`. Browse the available identifiers on the [OpenRouter Models](https://openrouter.ai/models) page.

```text
CineMatch first-run setup
Create an API key at https://openrouter.ai/settings/keys
OpenRouter API key (input is hidden):
OpenRouter model [openai/gpt-4o]:
```

The configuration is saved in CineMatch's operating-system user data directory and reused on later runs. The SQLite database is stored there as well. To change the API key or model later, run:

```powershell
cinematch --configure
```

You can also create a `.env` file in the directory where you run CineMatch. This local configuration takes precedence:

```env
OPENROUTER_API_KEY=pypi_user_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o
```

Run `cinematch --help` to select a host, port, or custom data directory.

### Build and publish a release

```powershell
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

PyPI accepts an API token as the username `__token__`; use the full token, including its `pypi-` prefix, as the password. Increase the version in `pyproject.toml` before every later release because PyPI release files cannot be replaced.

---

## Architecture & Design Documents

All architectural choices and domain terminology are formally documented:
- [`CONTEXT.md`](./CONTEXT.md) — Ubiquitous language and domain glossary.
- [`docs/adr/0001-hybrid-rag-taste-dossier.md`](./docs/adr/0001-hybrid-rag-taste-dossier.md) — Dual-layer RAG & Taste Dossier architecture.
- [`docs/adr/0002-six-point-forced-choice-rating-scale.md`](./docs/adr/0002-six-point-forced-choice-rating-scale.md) — 1–6 discrete forced-choice rating scale.
- [`docs/adr/0003-hybrid-catalog-orchestrator.md`](./docs/adr/0003-hybrid-catalog-orchestrator.md) — Local vector search + dynamic TMDB discovery fallback.
- [`docs/adr/0004-in-process-sqlite-vectors.md`](./docs/adr/0004-in-process-sqlite-vectors.md) — Zero-dependency in-process SQLite BLOB vector storage.
- [`docs/adr/0005-dual-source-embedding-strategy.md`](./docs/adr/0005-dual-source-embedding-strategy.md) — OpenRouter with offline local embedding toggle.
- [`docs/adr/0006-bundled-curated-starter-catalog.md`](./docs/adr/0006-bundled-curated-starter-catalog.md) — Bundled curated starter dataset.
- [`docs/adr/0007-compatible-embeddings.md`](./docs/adr/0007-compatible-embeddings.md) — Embedding compatibility, batched rebuilds, and lightweight local matching.

---

## Verification Tests

Install test dependencies and run the suite. Tests use a temporary database and disable external APIs, leaving your library untouched:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest -q
```
