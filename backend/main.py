import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_chain import RoadSafetyRAG
from rag_config import PROJECT_DIR, get_settings, get_settings_for_env_file


app = FastAPI(
    title="DALIL Road Safety Guide API",
    description="FastAPI + LangChain RAG chatbot over the Saudi road safety knowledge base.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


EMBEDDING_PROFILES = {
    "gemini-embedding-001": {
        "env_file": PROJECT_DIR / "comparison_versions" / "gemini-embedding-001" / "settings.env",
        "label": "Gemini Embedding 001",
    },
    "text-multilingual-embedding-002": {
        "env_file": PROJECT_DIR
        / "comparison_versions"
        / "text-multilingual-embedding-002"
        / "settings.env",
        "label": "Text Multilingual Embedding 002",
    },
}

DEFAULT_EMBEDDING_PROFILE = "gemini-embedding-001"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    chat_history: list["ChatMessage"] = Field(default_factory=list)
    embedding_profile: str = Field(default=DEFAULT_EMBEDDING_PROFILE)


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
    embedding_profile: str


def normalize_embedding_profile(profile: str | None) -> str:
    profile_name = (profile or DEFAULT_EMBEDDING_PROFILE).strip()
    return profile_name if profile_name in EMBEDDING_PROFILES else DEFAULT_EMBEDDING_PROFILE


def profile_settings(profile: str) -> dict[str, Any]:
    return EMBEDDING_PROFILES[normalize_embedding_profile(profile)]


@lru_cache(maxsize=len(EMBEDDING_PROFILES))
def get_rag(profile: str = DEFAULT_EMBEDDING_PROFILE) -> RoadSafetyRAG:
    profile_name = normalize_embedding_profile(profile)
    settings = get_settings_for_env_file(profile_settings(profile_name)["env_file"])
    return RoadSafetyRAG(settings)


@app.on_event("startup")
def warm_up_rag() -> None:
    rag = get_rag(DEFAULT_EMBEDDING_PROFILE)
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
        "default_embedding_profile": DEFAULT_EMBEDDING_PROFILE,
        "embedding_profiles": list(EMBEDDING_PROFILES.keys()),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        profile = normalize_embedding_profile(request.embedding_profile)
        result = get_rag(profile).ask(
            request.question,
            top_k=request.top_k,
            chat_history=[message.model_dump() for message in request.chat_history],
        )
        result["embedding_profile"] = profile
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(**result)


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    def generate():
        try:
            profile = normalize_embedding_profile(request.embedding_profile)
            events = get_rag(profile).ask_stream(
                request.question,
                top_k=request.top_k,
                chat_history=[message.model_dump() for message in request.chat_history],
            )
            for event in events:
                if event.get("type") in {"metadata", "done"}:
                    event["embedding_profile"] = profile
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
