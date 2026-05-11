from __future__ import annotations

import argparse

from tqdm import tqdm

from simple_rag.config import load_settings
from simple_rag.data import iter_clean_records, load_bkai_dataset
from simple_rag.embeddings import EmbeddingModel
from simple_rag.qdrant_store import QdrantStore


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed BKAI_RAG contexts and upload them to Qdrant.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to train or all splits.")
    parser.add_argument("--limit", type=int, default=None, help="Limit records for quick tests.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding/upsert batch size.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection first.")
    args = parser.parse_args()

    settings = load_settings()
    batch_size = args.batch_size or settings.batch_size

    print(f"Loading dataset {settings.dataset_name}/{settings.dataset_config}...")
    dataset = load_bkai_dataset(settings.dataset_name, settings.dataset_config, split=args.split)
    records = list(iter_clean_records(dataset, limit=args.limit))
    if not records:
        raise RuntimeError("No valid records found. Check dataset columns: question, answer, context.")

    print(f"Loading embedding model: {settings.embedding_model}")
    embedder = EmbeddingModel(settings.embedding_model)

    print(f"Connecting to Qdrant: {settings.qdrant_url}")
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.collection_name)
    store.ensure_collection(embedder.dimension, recreate=args.recreate)

    print(f"Indexing {len(records)} unique contexts into collection '{settings.collection_name}'...")
    for batch in tqdm(list(batched(records, batch_size))):
        vectors = embedder.encode([record.context for record in batch], batch_size=batch_size)
        store.upsert_records(batch, vectors.astype(float).tolist())

    print(f"Done. Qdrant collection count: {store.count()}")


if __name__ == "__main__":
    main()
