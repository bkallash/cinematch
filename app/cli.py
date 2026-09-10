"""Command-line entry point for an installed CineMatch application."""

import argparse
import getpass
import os
import sys
from pathlib import Path

from dotenv import load_dotenv, set_key

DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"


def default_data_dir() -> Path:
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return Path(root) / "CineMatch" if root else Path.home() / "CineMatch"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "CineMatch"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "cinematch"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CineMatch web application.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on.")
    parser.add_argument("--port", type=int, help="Port to listen on (default: APP_PORT or 8000).")
    parser.add_argument("--reload", action="store_true", help="Reload when source files change.")
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Prompt for the OpenRouter API key and model before starting.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory for cinematch.db (default: the operating system's user data directory).",
    )
    return parser


def openrouter_is_configured() -> bool:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", "").strip()
    return bool(api_key and not api_key.startswith("your_") and model)


def configure_openrouter(env_path: Path) -> None:
    if not sys.stdin.isatty():
        raise SystemExit(
            "OpenRouter is not configured. Run 'cinematch --configure' in an interactive "
            "terminal, or set OPENROUTER_API_KEY and OPENROUTER_MODEL."
        )

    print("\nCineMatch first-run setup")
    print("Create an API key at https://openrouter.ai/settings/keys")
    api_key = getpass.getpass("OpenRouter API key (input is hidden): ").strip()
    if not api_key:
        raise SystemExit("An OpenRouter API key is required.")

    current_model = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    model = input(f"OpenRouter model [{current_model}]: ").strip() or current_model

    env_path.touch(exist_ok=True)
    set_key(env_path, "OPENROUTER_API_KEY", api_key)
    set_key(env_path, "OPENROUTER_MODEL", model)
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENROUTER_MODEL"] = model
    print(f"Configuration saved to {env_path}\n")


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    user_env_path = data_dir / ".env"

    # A project-specific .env takes precedence over the saved user configuration.
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(user_env_path)
    if args.configure or not openrouter_is_configured():
        configure_openrouter(user_env_path)

    os.environ.setdefault("DATABASE_PATH", str(data_dir / "cinematch.db"))

    from app.config import settings
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port or settings.APP_PORT,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
