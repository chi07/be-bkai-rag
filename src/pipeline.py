from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.embeddings import EmbeddingModel
from src.generator import Generator
from src.qdrant_store import QdrantStore
from src.reranker import Reranker


@dataclass(frozen=True)
class RagResponse:
    answer: str
    contexts: list[dict]


class RagPipeline:
    def __init__(self, settings: Settings, preload_reranker: bool = False) -> None:
        self.settings = settings
        self.embedder = EmbeddingModel(settings.embedding_model, max_seq_length=settings.max_seq_length)
        self.store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.collection_name)
        self.generator = Generator(
            provider=settings.generator_provider,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        self.reranker = None
        if settings.rerank_enabled and preload_reranker:
            self._load_reranker()

    def _load_reranker(self) -> Reranker:
        if self.reranker is None:
            self.reranker = Reranker(
                self.settings.reranker_model,
                max_length=self.settings.rerank_max_length,
            )
        return self.reranker

    def retrieve(self, question: str, retrieve_k: int = 20) -> list[dict]:
        print("RAG stage: embedding query", flush=True)
        query_vector = self.embedder.encode_one(question)
        print("RAG stage: searching qdrant", flush=True)
        return self.store.search(query_vector, top_k=retrieve_k)

    def rerank(self, question: str, retrieved: list[dict], rerank_k: int = 5) -> list[dict]:
        if not self.settings.rerank_enabled:
            print("RAG stage: rerank disabled", flush=True)
            return retrieved[:rerank_k]
        print("RAG stage: reranking contexts", flush=True)
        reranker = self._load_reranker()
        return reranker.rerank(question, retrieved, top_k=rerank_k)

    def answer(self, question: str, retrieve_k: int = 20, rerank_k: int = 5) -> RagResponse:
        retrieved = self.retrieve(question, retrieve_k=retrieve_k)
        final_contexts = self.rerank(question, retrieved, rerank_k=rerank_k)
        context_texts = [item["payload"].get("context", "") for item in final_contexts]
        print("RAG stage: generating answer", flush=True)
        answer = self.generator.generate(question, context_texts)
        print("RAG stage: done", flush=True)
        return RagResponse(answer=answer, contexts=final_contexts)
