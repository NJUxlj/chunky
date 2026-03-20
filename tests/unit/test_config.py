"""Tests for chunky.config.settings."""

from __future__ import annotations

import yaml

from chunky.config.settings import (
    ChunkyConfig,
    EmbeddingConfig,
    LLMConfig,
    MilvusConfig,
    RerankerConfig,
    load_config,
    save_config,
)


# ── ChunkyConfig creation with defaults ──────────────────────────


class TestChunkyConfigDefaults:
    def test_default_config_has_expected_types(self):
        cfg = ChunkyConfig()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.reranker, RerankerConfig)
        assert isinstance(cfg.milvus, MilvusConfig)

    def test_default_test_mode_is_false(self):
        cfg = ChunkyConfig()
        assert cfg.test_mode is False

    def test_default_llm_values(self):
        cfg = ChunkyConfig()
        assert cfg.llm.api_base == ""
        assert cfg.llm.api_key == ""
        assert cfg.llm.model == ""
        assert cfg.llm.api_type == "openai"
        assert cfg.llm.max_tokens == 1024
        assert cfg.llm.temperature == 0.3

    def test_default_embedding_values(self):
        cfg = ChunkyConfig()
        assert cfg.embedding.model_name == "BAAI/bge-small-zh-v1.5"
        assert cfg.embedding.device == "cpu"
        assert cfg.embedding.batch_size == 32

    def test_default_milvus_values(self):
        cfg = ChunkyConfig()
        assert cfg.milvus.uri == "milvus_lite.db"
        assert cfg.milvus.use_lite is True
        assert cfg.milvus.default_collection == "chunky_default"
        assert cfg.milvus.dim == 512

    def test_default_reranker_values(self):
        cfg = ChunkyConfig()
        assert cfg.reranker.model_name == ""
        assert cfg.reranker.device == "cpu"


# ── to_dict / from_dict round-trip ───────────────────────────────


class TestConfigRoundTrip:
    def test_round_trip_default_config(self):
        original = ChunkyConfig()
        data = original.to_dict()
        restored = ChunkyConfig.from_dict(data)

        assert restored.test_mode == original.test_mode
        assert restored.llm.api_base == original.llm.api_base
        assert restored.llm.model == original.llm.model
        assert restored.embedding.model_name == original.embedding.model_name
        assert restored.milvus.uri == original.milvus.uri

    def test_round_trip_custom_config(self):
        original = ChunkyConfig(
            llm=LLMConfig(api_base="http://localhost:8000", model="my-model", api_key="sk-123"),
            embedding=EmbeddingConfig(model_name="custom-emb", device="cuda", batch_size=64),
            milvus=MilvusConfig(uri="localhost:19530", use_lite=False, default_collection="my_col", dim=768),
            test_mode=True,
        )
        data = original.to_dict()
        restored = ChunkyConfig.from_dict(data)

        assert restored.llm.api_base == "http://localhost:8000"
        assert restored.llm.model == "my-model"
        assert restored.llm.api_key == "sk-123"
        assert restored.embedding.model_name == "custom-emb"
        assert restored.embedding.device == "cuda"
        assert restored.embedding.batch_size == 64
        assert restored.milvus.uri == "localhost:19530"
        assert restored.milvus.use_lite is False
        assert restored.milvus.default_collection == "my_col"
        assert restored.milvus.dim == 768
        assert restored.test_mode is True

    def test_to_dict_returns_plain_dict(self):
        cfg = ChunkyConfig()
        data = cfg.to_dict()
        assert isinstance(data, dict)
        assert "llm" in data
        assert "embedding" in data
        assert "reranker" in data
        assert "milvus" in data
        assert "test_mode" in data

    def test_from_dict_with_empty_dict(self):
        cfg = ChunkyConfig.from_dict({})
        assert cfg.test_mode is False
        assert cfg.llm.api_type == "openai"

    def test_from_dict_with_partial_data(self):
        cfg = ChunkyConfig.from_dict({"test_mode": True, "llm": {"model": "gpt-4"}})
        assert cfg.test_mode is True
        assert cfg.llm.model == "gpt-4"
        assert cfg.llm.api_base == ""  # default


# ── save_config / load_config with tmp_path ──────────────────────


class TestConfigPersistence:
    def test_save_and_load_config(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_dir = tmp_path

        # Patch the module-level constants so we don't touch the real home dir
        import chunky.config.settings as settings_mod

        monkeypatch.setattr(settings_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)

        original = ChunkyConfig(
            llm=LLMConfig(api_base="http://test:8000", model="test-model"),
            test_mode=True,
        )
        save_config(original)

        assert config_file.exists()

        loaded = load_config()
        assert loaded.llm.api_base == "http://test:8000"
        assert loaded.llm.model == "test-model"
        assert loaded.test_mode is True

    def test_load_config_returns_defaults_when_file_missing(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        monkeypatch.setattr(settings_mod, "CONFIG_FILE", tmp_path / "nonexistent.yaml")

        cfg = load_config()
        assert cfg.test_mode is False
        assert cfg.llm.api_type == "openai"

    def test_saved_file_is_valid_yaml(self, tmp_path, monkeypatch):
        import chunky.config.settings as settings_mod

        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(settings_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(settings_mod, "CONFIG_FILE", config_file)

        save_config(ChunkyConfig())

        with open(config_file) as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        assert "llm" in data
        assert "milvus" in data
