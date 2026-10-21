"""Tests for reading the settings."""
import importlib

import app.config


def test_a_setting_missing_from_the_env_file_falls_back_to_the_example(monkeypatch):
    monkeypatch.delenv("MAX_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv("TOOL_TIMEOUT_SECONDS", raising=False)
    reloaded = importlib.reload(app.config)
    assert reloaded.settings.max_tool_rounds == 4
    assert reloaded.settings.tool_timeout_seconds == 10.0


def test_a_value_you_set_wins_over_the_example(monkeypatch):
    monkeypatch.setenv("MAX_TOOL_ROUNDS", "2")
    assert importlib.reload(app.config).settings.max_tool_rounds == 2
    monkeypatch.delenv("MAX_TOOL_ROUNDS")
    importlib.reload(app.config)  # leave the module as the other tests expect it
