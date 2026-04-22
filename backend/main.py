import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_chain import RoadSafetyRAG
from rag_config import get_settings


app = FastAPI(
    title="DALIL Road Safety Guide API",
    description="FastAPI + LangChain RAG chatbot over the Saudi road safety knowledge base.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    chat_history: list["ChatMessage"] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    is_fallback: bool
    language: str
    intent: str
    intent_detail: str | None = None
    answer_intent: str | None = None
    source_route: str | None = None
    needs_clarification: bool = False
    rewritten_query: str | None = None
    followup_topic: str | None = None
    followup_aspect: str | None = None
    used_rag: bool
    suggested_questions: list[str]
    model: str
    embedding_model: str
    embedding_dimensions: int


@lru_cache(maxsize=1)
def get_rag() -> RoadSafetyRAG:
    return RoadSafetyRAG(get_settings())


@app.on_event("startup")
def warm_up_rag() -> None:
    rag = get_rag()
    _ = rag.vector_store
    _ = rag.llm


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    index_path = Path(settings.vector_store_path) / "index.faiss"
    pkl_path = Path(settings.vector_store_path) / "index.pkl"
    return {
        "status": "ok",
        "knowledge_base_exists": settings.knowledge_base_path.exists(),
        "vector_store_exists": index_path.exists() and pkl_path.exists(),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "using_vertexai": settings.use_vertexai,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = get_rag().ask(
            request.question,
            top_k=request.top_k,
            chat_history=[message.model_dump() for message in request.chat_history],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(**result)


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    def generate():
        try:
            events = get_rag().ask_stream(
                request.question,
                top_k=request.top_k,
                chat_history=[message.model_dump() for message in request.chat_history],
            )
            for event in events:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except FileNotFoundError as exc:
            yield json.dumps(
                {"type": "error", "detail": str(exc)}, ensure_ascii=False
            ) + "\n"
        except Exception as exc:
            yield json.dumps(
                {"type": "error", "detail": str(exc)}, ensure_ascii=False
            ) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
