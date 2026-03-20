"""Configuration management for chunky using pydantic and pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


CONFIG_DIR = Path.home() / ".config" / "chunky"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


class LLMConfig(BaseModel):
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    api_type: str = "openai"  # "openai" or "vllm"
    max_tokens: int = 1024
    temperature: float = 0.3


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-zh-v1.5"
    api_base: str = ""  # Optional API base URL for embedding service
    api_key: str = ""    # Optional API key for embedding service
    device: str = "cpu"
    batch_size: int = 32


class RerankerConfig(BaseModel):
    model_name: str = ""
    device: str = "cpu"


class MilvusConfig(BaseModel):
    uri: str = "milvus_lite.db"  # file path for milvus-lite, or host:port for milvus
    use_lite: bool = True
    default_collection: str = "chunky_default"
    dim: int = 512


class ChromaConfig(BaseModel):
    persist_directory: str = "~/.chunky/chroma_db"
    default_collection: str = "chunky_default"
    dim: int = 384


class ChunkyConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    vector_store_type: str = "chroma"  # "chroma" or "milvus"
    test_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkyConfig:
        return cls.model_validate(data)


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> ChunkyConfig:
    if not CONFIG_FILE.exists():
        return ChunkyConfig()
    with open(CONFIG_FILE) as f:
        data = yaml.safe_load(f) or {}
    return ChunkyConfig.from_dict(data)


def save_config(config: ChunkyConfig) -> None:
    ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
