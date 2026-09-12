"""Installed CLI configuration must be isolated from package files."""

import os
from unittest.mock import Mock

from app import cli


def test_model_override_and_relative_data_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "saved/model")
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    monkeypatch.setattr("sys.argv", ["cinematch", "--model", "chosen/model", "--data-dir", "library"])
    run = Mock()
    monkeypatch.setattr("uvicorn.run", run)
    cli.main()
    assert os.environ["OPENROUTER_MODEL"] == "chosen/model"
    assert os.environ["DATABASE_PATH"] == str(tmp_path / "library" / "cinematch.db")
    assert run.call_args.kwargs["host"] == "127.0.0.1"


def test_setup_saves_key_and_model(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda _: "test-key")
    monkeypatch.setattr("builtins.input", lambda _: "chosen/model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    path = tmp_path / ".env"
    cli.configure_openrouter(path)
    assert cli.openrouter_is_configured()
    assert cli.load_dotenv(path)
    assert "chosen/model" in path.read_text()


def test_missing_configuration_fails_without_prompt(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="interactive"):
        cli.configure_openrouter(tmp_path / ".env")
