from __future__ import annotations

import argparse

from tqdm import tqdm

from simple_rag.config import load_settings
from simple_rag.data import iter_eval_rows, load_bkai_dataset
from simple_rag.embeddings import EmbeddingModel
from simple_rag.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall@k against each row's source context.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to train or all splits.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = load_settings()
    dataset = load_bkai_dataset(settings.dataset_name, settings.dataset_config, split=args.split)
    rows = list(iter_eval_rows(dataset, limit=args.limit))

    embedder = EmbeddingModel(settings.embedding_model)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.collection_name)

    hits = 0
    reciprocal_ranks: list[float] = []

    for row in tqdm(rows):
        query_vector = embedder.encode_one(str(row["question"]))
        results = store.search(query_vector, top_k=args.top_k)
        hashes = [item["payload"].get("context_hash") for item in results]
        expected_hashes = set(row["context_hashes"])

        if expected_hashes.intersection(hashes):
            hits += 1
            first_rank = min(index for index, value in enumerate(hashes) if value in expected_hashes) + 1
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(rows)
    recall = hits / total if total else 0.0
    mrr = sum(reciprocal_ranks) / total if total else 0.0

    print(f"evaluated={total}")
    print(f"recall@{args.top_k}={recall:.4f}")
    print(f"mrr@{args.top_k}={mrr:.4f}")


if __name__ == "__main__":
    main()
