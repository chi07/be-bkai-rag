from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models

from simple_rag.data import RagRecord


class QdrantStore:
    def __init__(self, url: str, api_key: str | None, collection_name: str) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if recreate and self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
            existing.remove(self.collection_name)

        if self.collection_name in existing:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_records(self, records: Sequence[RagRecord], vectors: Sequence[Sequence[float]]) -> None:
        points = [
            models.PointStruct(
                id=record.doc_id,
                vector=vector,
                payload={
                    "doc_id": record.doc_id,
                    "context_hash": record.context_hash,
                    "context": record.context,
                    "sample_question": record.question,
                    "sample_answer": record.answer,
                    "source_row_id": record.source_row_id,
                    "source_context_index": record.source_context_index,
                },
            )
            for record, vector in zip(records, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[dict]:
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = result.points
        except AttributeError:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload or {},
            }
            for point in points
        ]

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection_name, exact=True)
        return result.count
