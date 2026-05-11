from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    dataset_name: str = "sailor2/Vietnamese_RAG"
    dataset_config: str = "BKAI_RAG"
    collection_name: str = "vietnamese_rag_bkai"
    embedding_model: str = "BAAI/bge-m3"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    batch_size: int = 64


def load_settings() -> Settings:
    return Settings(
        dataset_name=os.getenv("RAG_DATASET_NAME", Settings.dataset_name),
        dataset_config=os.getenv("RAG_DATASET_CONFIG", Settings.dataset_config),
        collection_name=os.getenv("RAG_COLLECTION", Settings.collection_name),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", Settings.embedding_model),
        qdrant_url=os.getenv("QDRANT_URL", Settings.qdrant_url),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        batch_size=int(os.getenv("RAG_BATCH_SIZE", Settings.batch_size)),
    )
