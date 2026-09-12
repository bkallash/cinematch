# CineMatch

A personal movie and TV recommendation app that runs in your browser. Rate titles, keep a watchlist, and chat with an assistant that uses your preferences. Bring your own OpenRouter API key and choose your model.

## Install and run

Requires Python 3.10 or newer.

```sh
pip install cinematch
cinematch
```

On first launch, enter your [OpenRouter API key](https://openrouter.ai/settings/keys) (hidden as you type) and a model ID. The default is `openai/gpt-4o-mini`. Choose a chat model that supports JSON object responses.

Open **http://127.0.0.1:8000** after startup. Press Ctrl+C to stop.

```sh
cinematch --configure                # Save a different key or model
cinematch --model openai/gpt-4o-mini  # Select a model for this run
cinematch --port 8080
cinematch --data-dir ./my-library
```

## Configuration

For noninteractive startup, set both `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` in your environment or a `.env` file in your working directory. See [.env.example](.env.example) for optional settings. Environment variables take precedence over the working directory's `.env`, then saved configuration. `--model` overrides the model for the current run.

| Setting | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Your OpenRouter key |
| `OPENROUTER_MODEL` | Chat and recommendation model ID |
| `EMBEDDING_PROVIDER` | `openrouter` (default) or `local` |
| `EMBEDDING_MODEL` | Defaults to `openai/text-embedding-3-small` |
| `EMBEDDING_DIM` | Embedding dimensions; defaults to `1536` |
| `TMDB_API_KEY` | Optional TMDB key for live catalog lookup |
| `DATABASE_PATH` | Override the SQLite file location |
| `APP_PORT` | Port; defaults to `8000` |

A starter catalog is bundled, so TMDB credentials are optional. Embeddings use a separate OpenRouter model; `EMBEDDING_PROVIDER=local` uses local token hashing instead. OpenRouter calls use your account and may incur charges. Prompts and recommendation context are sent to the configured provider.

The CLI saves its `.env` and SQLite library in the following directory, unless you specify `--data-dir`:

| Platform | Default directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\CineMatch`, falling back to `%APPDATA%\CineMatch` |
| macOS | `~/Library/Application Support/CineMatch` |
| Linux | `$XDG_DATA_HOME/cinematch` or `~/.local/share/cinematch` |

API keys are saved as plaintext in the local configuration file. Keep it private. This is a personal app without user authentication; it listens on localhost by default.

## Development

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
cinematch --reload
```

FastAPI serves Jinja templates and static assets. SQLite stores ratings, watchlists, and recommendation data. The package includes its starter catalog, templates, CSS, and placeholder image.

## Build and publish

The distribution name and version are in `pyproject.toml`. You must own the PyPI project, and every release needs a new version.

```sh
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/cinematch-0.1.1*
```

Install the wheel in a fresh virtual environment and run `cinematch` outside the checkout. Then upload this release:

```sh
python -m twine upload dist/cinematch-0.1.1-py3-none-any.whl dist/cinematch-0.1.1.tar.gz
```

Use your PyPI API token when prompted. See the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) for account setup and TestPyPI instructions.

## GitHub

Publish the release commit to a public repository in your GitHub account. Review the files and history before pushing. Local keys, databases, build output, and legacy documentation are ignored and excluded from release artifacts.

There is currently no license file. Choose a license before advertising reuse rights.
