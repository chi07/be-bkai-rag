from __future__ import annotations

import argparse

from src.config import load_settings
from src.embeddings import EmbeddingModel
from src.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a user query and retrieve contexts from Qdrant.")
    parser.add_argument("question", help="User question.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = load_settings()
    embedder = EmbeddingModel(settings.embedding_model, max_seq_length=settings.max_seq_length)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.collection_name)

    query_vector = embedder.encode_one(args.question)
    results = store.search(query_vector, top_k=args.top_k)

    for rank, item in enumerate(results, start=1):
        payload = item["payload"]
        print(f"\n#{rank} score={item['score']:.4f} doc_id={payload.get('doc_id')}")
        print(payload.get("context", ""))


if __name__ == "__main__":
    main()
