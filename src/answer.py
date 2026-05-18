from __future__ import annotations

import argparse

from src.config import load_settings
from src.pipeline import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieve -> rerank -> generate for a question.")
    parser.add_argument("question", help="User question.")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--rerank-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--show-contexts", action="store_true")
    args = parser.parse_args()

    settings = load_settings()

    print(f"Embedding query with: {settings.embedding_model}")
    if settings.rerank_enabled and not args.no_rerank:
        print(f"Reranking with: {settings.reranker_model}")
    else:
        print("Rerank disabled; using vector search order.")
    print(f"Generating answer with provider: {settings.generator_provider}")

    pipeline = RagPipeline(settings, preload_reranker=not args.no_rerank)
    result = pipeline.answer(args.question, retrieve_k=args.retrieve_k, rerank_k=args.rerank_k)

    print("\nAnswer:")
    print(result.answer)

    if args.show_contexts:
        print("\nContexts:")
        for rank, item in enumerate(result.contexts, start=1):
            payload = item["payload"]
            vector_score = item.get("score", 0.0)
            rerank_score = item.get("rerank_score")
            score_text = f"vector={vector_score:.4f}"
            if rerank_score is not None:
                score_text += f" rerank={rerank_score:.4f}"
            print(f"\n#{rank} {score_text} doc_id={payload.get('doc_id')}")
            print(payload.get("context", ""))


if __name__ == "__main__":
    main()
