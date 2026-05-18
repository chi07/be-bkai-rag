from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, question: str, results: list[dict], top_k: int = 5) -> list[dict]:
        if not results:
            return []

        pairs = [(question, item["payload"].get("context", "")) for item in results]
        scores = self.model.predict(pairs)

        reranked = []
        for item, score in zip(results, scores, strict=True):
            updated = dict(item)
            updated["rerank_score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k]
