import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google.oauth2 import service_account
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent if BACKEND_DIR.name == "backend" else BACKEND_DIR


def resolve_env_file(env_file: str | Path | None = None) -> Path:
    custom_env = env_file or os.getenv("RAG_ENV_FILE")
    if custom_env:
        custom_path = Path(custom_env).expanduser()
        if not custom_path.is_absolute():
            custom_path = (PROJECT_DIR / custom_path).resolve()
        return custom_path
    return BACKEND_DIR / ".env" if (BACKEND_DIR / ".env").exists() else PROJECT_DIR / ".env"


ENV_FILE = resolve_env_file()


def resolve_project_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    google_genai_use_vertexai: bool = Field(
        default=False, validation_alias="GOOGLE_GENAI_USE_VERTEXAI"
    )
    google_application_credentials: Path | None = Field(
        default=None, validation_alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    google_cloud_project: str | None = Field(
        default=None, validation_alias="GOOGLE_CLOUD_PROJECT"
    )
    google_cloud_location: str = Field(
        default="us-central1", validation_alias="GOOGLE_CLOUD_LOCATION"
    )

    knowledge_base_path: Path = Field(
        default=PROJECT_DIR / "data" / "saudi_road_safety_kb.json",
        validation_alias="KNOWLEDGE_BASE_PATH",
    )
    vector_store_path: Path = Field(
        default=PROJECT_DIR / "data" / "vector_store",
        validation_alias="VECTOR_STORE_PATH",
    )

    embedding_model: str = Field(
        default="gemini-embedding-001", validation_alias="GEMINI_EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(
        default=3072, validation_alias="GEMINI_EMBEDDING_DIMENSIONS"
    )
    llm_model: str = Field(
        default="gemini-3.1-pro-preview", validation_alias="GEMINI_LLM_MODEL"
    )
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = Field(
        default="low", validation_alias="GEMINI_THINKING_LEVEL"
    )

    retriever_top_k: int = Field(default=6, validation_alias="RETRIEVER_TOP_K")
    llm_temperature: float = Field(default=0.2, validation_alias="LLM_TEMPERATURE")

    @property
    def api_key(self) -> str | None:
        return self.google_api_key or self.gemini_api_key

    @property
    def use_vertexai(self) -> bool:
        return self.google_genai_use_vertexai or (
            self.google_application_credentials is not None and self.api_key is None
        )


def get_settings() -> Settings:
    return get_settings_for_env_file()


def get_settings_for_env_file(env_file: str | Path | None = None) -> Settings:
    resolved_env_file = resolve_env_file(env_file)
    load_dotenv(resolved_env_file, override=True)
    settings = Settings(
        _env_file=resolved_env_file,
        _env_file_encoding="utf-8",
    )
    settings.google_application_credentials = resolve_project_path(
        settings.google_application_credentials
    )
    settings.knowledge_base_path = resolve_project_path(settings.knowledge_base_path)
    settings.vector_store_path = resolve_project_path(settings.vector_store_path)
    return settings


def load_google_credentials(settings: Settings):
    if not settings.google_application_credentials:
        return None

    credentials_path = settings.google_application_credentials.expanduser()
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Google credentials file was not found: {credentials_path}"
        )

    return service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
