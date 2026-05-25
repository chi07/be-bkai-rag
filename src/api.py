from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4
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
    session_id: str | None = None
    use_history: bool = True
    history_turns: int = Field(3, ge=0, le=10)


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


def recent_history(history: list[dict[str, str]], turns: int) -> list[dict[str, str]]:
    if turns <= 0:
        return []
    return history[-turns * 2 :]


def build_retrieval_question(question: str, history: list[dict[str, str]]) -> str:
    if not history:
        return question
    lines = []
    for turn in history:
        role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines + [f"Câu hỏi hiện tại: {question}"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.pipeline = RagPipeline(settings)
    app.state.sessions = {}
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
        session_id = request.session_id or str(uuid4())
        session_history = app.state.sessions.setdefault(session_id, [])
        history = recent_history(session_history, request.history_turns) if request.use_history else []
        retrieval_question = build_retrieval_question(request.question, history)

        result = app.state.pipeline.answer(
            request.question,
            retrieve_k=request.retrieve_k,
            rerank_k=request.rerank_k,
            history=history,
            retrieval_question=retrieval_question,
        )

        session_history.extend(
            [
                {"role": "user", "content": request.question},
                {"role": "assistant", "content": result.answer},
            ]
        )
        app.state.sessions[session_id] = session_history[-20:]

        response: dict[str, Any] = {"answer": result.answer, "session_id": session_id}
        if request.include_contexts:
            response["contexts"] = serialize_contexts(result.contexts)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    app.state.sessions.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}
