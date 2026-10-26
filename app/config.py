"""Validated application settings loaded from environment variables.

Values are read from the process environment, then `.env`, then `.env.local`
(later files override earlier ones; real environment variables override both).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_API_KEYS = 10
T = TypeVar("T")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ResearchFlow AI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # --- LLM provider (Gemini) -------------------------------------------------
    # Keys can be given as GOOGLE_API_KEY, GOOGLE_API_KEY_1..GOOGLE_API_KEY_10 and/or a
    # comma-separated GOOGLE_API_KEYS. They are merged (deduplicated, order kept) into
    # a rotating key pool; see app/llm/key_pool.py.
    google_api_key: SecretStr | None = None
    google_api_keys: str | None = None
    google_api_key_1: SecretStr | None = None
    google_api_key_2: SecretStr | None = None
    google_api_key_3: SecretStr | None = None
    google_api_key_4: SecretStr | None = None
    google_api_key_5: SecretStr | None = None
    google_api_key_6: SecretStr | None = None
    google_api_key_7: SecretStr | None = None
    google_api_key_8: SecretStr | None = None
    google_api_key_9: SecretStr | None = None
    google_api_key_10: SecretStr | None = None

    chat_model: str = "gemini-3.5-flash"
    # Comma-separated models tried (with every key) once the primary model's quota is
    # exhausted on all keys. Free-tier quotas are per project *and per model*.
    chat_fallback_models: str = ""
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    # Seconds an exhausted key is benched when the provider gives no retry delay.
    key_cooldown_seconds: float = Field(default=60.0, ge=0)
    # When all keys are cooling down, wait up to this long for one to recover.
    llm_max_wait_seconds: float = Field(default=65.0, ge=0, le=300)
    # Bench time when a daily quota is exhausted.
    key_daily_quota_cooldown_seconds: float = Field(default=3600.0, ge=0)
    # Re-ask the model this many times when it returns unparseable structured output.
    llm_invalid_output_retries: int = Field(default=1, ge=0, le=3)
    # Max keys tried for one call when failures are transient (5xx, overload, timeout).
    llm_max_transient_failovers: int = Field(default=3, ge=1, le=10)

    # --- Embeddings ---------------------------------------------------------------
    embedding_provider: Literal["huggingface", "hashing"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # --- Storage --------------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    database_url: str | None = None
    checkpoint_db_path: Path | None = None
    upload_dir: Path | None = None
    vector_store_dir: Path | None = None
    key_pool_state_path: Path | None = None

    # --- Ingestion & retrieval -----------------------------------------------------------
    max_upload_size_mb: float = Field(default=20.0, gt=0)
    max_files_per_upload: int = Field(default=10, ge=1)
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    max_evidence_chunks: int = Field(default=8, ge=1, le=50)
    # Cosine-similarity floor for retrieved chunks (0 disables the filter).
    retrieval_min_score: float = Field(default=0.1, ge=-1.0, le=1.0)

    # --- Workflow -------------------------------------------------------------------------
    max_retries: int = Field(default=2, ge=0, le=5)
    max_question_chars: int = Field(default=2000, ge=10)
    max_history_turns: int = Field(default=6, ge=0)

    # --- API / UI -------------------------------------------------------------------------
    cors_origins: str = "http://localhost:8501"
    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: float = 180.0

    # --- Observability --------------------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "researchflow-ai"

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
        return level

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        data = self.data_dir
        if self.database_url is None:
            self.database_url = f"sqlite:///{data / 'researchflow.db'}"
        if self.checkpoint_db_path is None:
            self.checkpoint_db_path = data / "checkpoints.db"
        if self.upload_dir is None:
            self.upload_dir = data / "uploads"
        if self.vector_store_dir is None:
            self.vector_store_dir = data / "vector_store"
        if self.key_pool_state_path is None:
            self.key_pool_state_path = data / "key_pool_state.json"
        return self

    # ------------------------------------------------------------------------------------
    @property
    def api_keys(self) -> list[str]:
        """All configured Gemini keys, deduplicated, in priority order."""
        candidates: list[str] = []
        if self.google_api_key is not None:
            candidates.append(self.google_api_key.get_secret_value())
        for index in range(1, MAX_API_KEYS + 1):
            secret: SecretStr | None = getattr(self, f"google_api_key_{index}")
            if secret is not None:
                candidates.append(secret.get_secret_value())
        if self.google_api_keys:
            candidates.extend(self.google_api_keys.split(","))
        keys: list[str] = []
        for key in (candidate.strip() for candidate in candidates):
            if key and key not in keys:
                keys.append(key)
        return keys

    @property
    def chat_models(self) -> list[str]:
        """Primary model followed by fallbacks, deduplicated, in order."""
        models = [self.chat_model, *self.chat_fallback_models.split(",")]
        return list(dict.fromkeys(m.strip() for m in models if m.strip()))

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_keys)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # Typed accessors for the derived settings (always set by _derive_paths).
    @property
    def database_url_value(self) -> str:
        return _required(self.database_url, "DATABASE_URL")

    @property
    def upload_path(self) -> Path:
        return _required(self.upload_dir, "UPLOAD_DIR")

    @property
    def vector_store_path(self) -> Path:
        return _required(self.vector_store_dir, "VECTOR_STORE_DIR")

    @property
    def checkpoint_path(self) -> Path:
        return _required(self.checkpoint_db_path, "CHECKPOINT_DB_PATH")

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_path, self.vector_store_path):
            path.mkdir(parents=True, exist_ok=True)


def _required(value: T | None, name: str) -> T:
    if value is None:
        raise RuntimeError(f"Setting {name} was not initialised.")
    return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
