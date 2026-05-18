from __future__ import annotations

import argparse

from tqdm import tqdm

from src.config import load_settings
from src.data import iter_clean_records, load_bkai_dataset
from src.embeddings import EmbeddingModel
from src.qdrant_store import QdrantStore


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed BKAI_RAG contexts and upload them to Qdrant.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to train or all splits.")
    parser.add_argument("--limit", type=int, default=None, help="Limit records for quick tests.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding/upsert batch size.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Max token length before truncation.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection first.")
    args = parser.parse_args()

    settings = load_settings()
    batch_size = args.batch_size or settings.batch_size
    max_seq_length = args.max_seq_length or settings.max_seq_length

    print(f"Loading dataset {settings.dataset_name}/{settings.dataset_config}...")
    dataset = load_bkai_dataset(settings.dataset_name, settings.dataset_config, split=args.split)
    records = list(iter_clean_records(dataset, limit=args.limit))
    if not records:
        raise RuntimeError("No valid records found. Check dataset columns: question, answer, context.")

    print(f"Loading embedding model: {settings.embedding_model}")
    print(f"Embedding batch size: {batch_size}; max sequence length: {max_seq_length}")
    embedder = EmbeddingModel(settings.embedding_model, max_seq_length=max_seq_length)

    print(f"Connecting to Qdrant: {settings.qdrant_url}")
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.collection_name)
    store.ensure_collection(embedder.dimension, recreate=args.recreate)

    print(f"Indexing {len(records)} unique contexts into collection '{settings.collection_name}'...")
    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch in tqdm(batched(records, batch_size), total=total_batches):
        vectors = embedder.encode([record.context for record in batch], batch_size=batch_size)
        store.upsert_records(batch, vectors.astype(float).tolist())

    print(f"Done. Qdrant collection count: {store.count()}")


if __name__ == "__main__":
    main()
