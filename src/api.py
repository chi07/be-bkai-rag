from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_settings
from src.pipeline import RagPipeline


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retrieve_k: int = Field(20, ge=1, le=100)
    rerank_k: int = Field(5, ge=1, le=20)
    include_contexts: bool = False


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)


def serialize_contexts(contexts: list[dict]) -> list[dict[str, Any]]:
    serialized = []
    for rank, item in enumerate(contexts, start=1):
        payload = item.get("payload", {})
        serialized.append(
            {
                "rank": rank,
                "doc_id": payload.get("doc_id"),
                "source_row_id": payload.get("source_row_id"),
                "source_context_index": payload.get("source_context_index"),
                "vector_score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
                "context": payload.get("context"),
            }
        )
    return serialized


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.pipeline = RagPipeline(settings)
    yield


app = FastAPI(title="Simple Vietnamese RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    settings = app.state.settings
    return {
        "status": "ok",
        "collection": settings.collection_name,
        "embedding_model": settings.embedding_model,
        "generator_provider": settings.generator_provider,
        "llm_model": settings.llm_model,
        "rerank_enabled": str(settings.rerank_enabled).lower(),
    }


@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict[str, Any]:
    try:
        results = app.state.pipeline.retrieve(request.question, retrieve_k=request.top_k)
        return {"contexts": serialize_contexts(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, Any]:
    try:
        result = app.state.pipeline.answer(
            request.question,
            retrieve_k=request.retrieve_k,
            rerank_k=request.rerank_k,
        )
        response: dict[str, Any] = {"answer": result.answer}
        if request.include_contexts:
            response["contexts"] = serialize_contexts(result.contexts)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
