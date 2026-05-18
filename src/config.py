from __future__ import annotations

import os
from dataclasses import dataclass


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    dataset_name: str = "sailor2/Vietnamese_RAG"
    dataset_config: str = "BKAI_RAG"
    collection_name: str = "vietnamese_rag_bkai"
    embedding_model: str = "BAAI/bge-m3"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    batch_size: int = 4
    max_seq_length: int = 512
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True
    rerank_max_length: int = 512
    generator_provider: str = "extractive"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str | None = None
    llm_model: str = "llama3.1"


def load_settings() -> Settings:
    return Settings(
        dataset_name=os.getenv("RAG_DATASET_NAME", Settings.dataset_name),
        dataset_config=os.getenv("RAG_DATASET_CONFIG", Settings.dataset_config),
        collection_name=os.getenv("RAG_COLLECTION", Settings.collection_name),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", Settings.embedding_model),
        qdrant_url=os.getenv("QDRANT_URL", Settings.qdrant_url),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        batch_size=int(os.getenv("RAG_BATCH_SIZE", Settings.batch_size)),
        max_seq_length=int(os.getenv("RAG_MAX_SEQ_LENGTH", Settings.max_seq_length)),
        reranker_model=os.getenv("RAG_RERANKER_MODEL", Settings.reranker_model),
        rerank_enabled=env_bool("RAG_RERANK_ENABLED", Settings.rerank_enabled),
        rerank_max_length=int(os.getenv("RAG_RERANK_MAX_LENGTH", Settings.rerank_max_length)),
        generator_provider=os.getenv("RAG_GENERATOR_PROVIDER", Settings.generator_provider),
        llm_base_url=os.getenv("RAG_LLM_BASE_URL", Settings.llm_base_url),
        llm_api_key=os.getenv("RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        llm_model=os.getenv("RAG_LLM_MODEL", Settings.llm_model),
    )
