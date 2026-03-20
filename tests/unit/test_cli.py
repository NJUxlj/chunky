"""Tests for chunky.cli.main using click.testing.CliRunner."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from chunky.cli.main import cli
from chunky.config.settings import ChunkyConfig, save_config


# ── chunky init --test ───────────────────────────────────────────


class TestInitCommand:
    def test_init_test_mode(self, tmp_path, monkeypatch):
        """chunky init --test should complete successfully with prompted input."""
        import chunky.config.settings as settings_mod

        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", tmp_path / "config.yaml")

        runner = CliRunner()
        # Provide answers for the interactive prompts:
        # LLM: api_type, api_base, api_key, model
        # Milvus: use_lite, db_path, collection
        result = runner.invoke(
            cli,
            ["init", "--test"],
            input="openai\nhttps://api.openai.com/v1\nsk-test\ngpt-4o-mini\ny\nmilvus_lite.db\nchunky_default\n",
        )
        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Configuration saved" in result.output

    def test_init_test_mode_creates_config_file(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["init", "--test"],
            input="openai\nhttps://api.openai.com/v1\nsk-test\ngpt-4o-mini\ny\nmilvus_lite.db\ntest_col\n",
        )
        assert result.exit_code == 0, f"Output: {result.output}"
        assert config_file.exists()


# ── chunky models list ───────────────────────────────────────────


class TestModelsListCommand:
    def test_models_list_runs_without_error(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)

        # Pre-save a config so load_config() finds it
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)
        save_config(ChunkyConfig())

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "list"])
        assert result.exit_code == 0, f"Output: {result.output}"
        assert "LLM Configuration" in result.output

    def test_models_list_shows_default_values(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)
        save_config(ChunkyConfig())

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "list"])
        assert result.exit_code == 0
        # Should show default embedding model
        assert "bge-small-zh" in result.output or "not set" in result.output


# ── chunky milvus --collection ───────────────────────────────────


class TestMilvusCommand:
    def test_milvus_config_collection(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)
        save_config(ChunkyConfig())

        runner = CliRunner()
        result = runner.invoke(cli, ["milvus", "--collection", "test_col"])
        assert result.exit_code == 0, f"Output: {result.output}"
        assert "test_col" in result.output

    def test_milvus_without_args_shows_current(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)
        save_config(ChunkyConfig())

        runner = CliRunner()
        result = runner.invoke(cli, ["milvus"])
        assert result.exit_code == 0
        assert "chunky_default" in result.output

    def test_milvus_collection_persists(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)
        save_config(ChunkyConfig())

        runner = CliRunner()
        # Set the collection
        runner.invoke(cli, ["milvus", "--collection", "my_new_col"])
        # Read it back
        result = runner.invoke(cli, ["milvus"])
        assert result.exit_code == 0
        assert "my_new_col" in result.output


# ── CLI group ────────────────────────────────────────────────────


class TestCLIGroup:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "chunky" in result.output

    def test_models_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0

    def test_milvus_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["milvus", "--help"])
        assert result.exit_code == 0
