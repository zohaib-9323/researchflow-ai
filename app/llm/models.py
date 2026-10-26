"""Chat model construction and the structured-output gateway.

All provider-specific code lives here. `LLMGateway.structured()` runs a LangChain
`prompt | model.with_structured_output(schema)` chain through the rotating key pool,
validates the result and converts provider errors into domain errors.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar

import httpx
from google.genai.errors import APIError as GoogleSDKError
from langchain_core.exceptions import (
    ContextOverflowError,
    LangChainException,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.core.exceptions import InvalidModelOutputError, ProviderError
from app.core.logging import get_logger
from app.llm.key_pool import (
    INVALID_KEY_COOLDOWN_SECONDS,
    FailureKind,
    KeyFailure,
    KeyPool,
    PoolEntry,
    parse_retry_delay,
)

logger = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
ModelFactory = Callable[[str, str], BaseChatModel]  # (api_key, model) -> chat model

# Collects the models that produced each structured result during one request, so the
# response can report which (fallback) model answered. Set by the research service.
models_used_var: ContextVar[list[str] | None] = ContextVar("models_used", default=None)

_INVALID_KEY_MARKERS = ("api_key_invalid", "api key not valid", "api key expired")


class _InvalidOutputError(Exception):
    pass


def build_gemini_factory(settings: Settings) -> ModelFactory:
    """Return a factory creating one ChatGoogleGenerativeAI client per key and model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    def factory(api_key: str, model: str) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            # One attempt per key: on 429 we rotate keys instead of waiting.
            max_retries=1,
        )

    return factory


# Classic "AIza..." keys and the newer "AQ." key format.
_API_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|AQ\.[0-9A-Za-z_\-]{20,}")


def redact_secrets(text: str) -> str:
    """Remove anything that looks like a Google API key before logging."""
    return _API_KEY.sub("[redacted]", text)


def classify_provider_error(exc: BaseException, settings: Settings) -> KeyFailure | None:
    """Map a provider exception to a key-pool failure (None = not key related)."""
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, ModelRateLimitError):
        delay = parse_retry_delay(message)
        if "perday" in lowered.replace(" ", "") or "per day" in lowered:
            cooldown = max(delay or 0.0, settings.key_daily_quota_cooldown_seconds)
        else:
            cooldown = delay if delay is not None else settings.key_cooldown_seconds
        return KeyFailure(FailureKind.RATE_LIMITED, cooldown)
    if isinstance(exc, ModelAuthenticationError | ModelPermissionDeniedError) or (
        isinstance(exc, ModelInvalidRequestError)
        and any(marker in lowered for marker in _INVALID_KEY_MARKERS)
    ):
        return KeyFailure(FailureKind.INVALID_KEY, INVALID_KEY_COOLDOWN_SECONDS)
    if isinstance(exc, ContextOverflowError):
        return None
    if isinstance(
        exc,
        ModelAPIError | ModelTimeoutError | ModelConnectionError | httpx.TransportError,
    ):
        return KeyFailure(FailureKind.TRANSIENT)
    return None


class LLMGateway:
    def __init__(
        self,
        settings: Settings,
        pool: KeyPool,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self._factory = model_factory
        self._models: dict[tuple[str, str], BaseChatModel] = {}
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return len(self.pool) > 0

    def _model(self, entry: PoolEntry) -> BaseChatModel:
        cache_key = (entry.key, entry.model)
        with self._lock:
            if cache_key not in self._models:
                if self._factory is None:
                    self._factory = build_gemini_factory(self.settings)
                self._models[cache_key] = self._factory(entry.key, entry.model)
            return self._models[cache_key]

    def structured(
        self,
        prompt: ChatPromptTemplate,
        schema: type[SchemaT],
        inputs: dict[str, Any],
        *,
        task: str,
    ) -> SchemaT:
        """Invoke `prompt | model.with_structured_output(schema)` with key failover.

        Unparseable output is re-requested up to `llm_invalid_output_retries` times,
        then raised as InvalidModelOutputError.
        """

        def operation(entry: PoolEntry) -> SchemaT:
            chain = prompt | self._model(entry).with_structured_output(schema, include_raw=True)
            result = chain.invoke(inputs, config={"run_name": task, "tags": [task]})
            if not isinstance(result, dict):
                raise _InvalidOutputError(task)
            parsed = result.get("parsed")
            if result.get("parsing_error") is not None or parsed is None:
                raise _InvalidOutputError(task)
            try:
                validated = schema.model_validate(parsed)
            except ValidationError as exc:
                raise _InvalidOutputError(task) from exc
            if (used := models_used_var.get()) is not None:
                used.append(entry.model)
            return validated

        def classify(exc: BaseException) -> KeyFailure | None:
            failure = classify_provider_error(exc, self.settings)
            if failure is not None:
                logger.warning(
                    "llm_provider_failure",
                    extra={
                        "task": task,
                        "error_type": type(exc).__name__,
                        "error": redact_secrets(str(exc))[:300],
                    },
                )
            return failure

        attempts = self.settings.llm_invalid_output_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self.pool.execute(
                    operation,
                    classify,
                    max_transient_failures=self.settings.llm_max_transient_failovers,
                )
            except _InvalidOutputError:
                logger.warning("llm_invalid_output", extra={"task": task, "attempt": attempt})
            except ProviderError:
                raise
            except (ModelNotFoundError, ModelInvalidRequestError, ContextOverflowError) as exc:
                logger.error("llm_request_rejected", extra={"task": task})
                raise ProviderError(
                    "The language model rejected the request. Check CHAT_MODEL and input size.",
                    details={"error_type": type(exc).__name__},
                ) from exc
            except (LangChainException, GoogleSDKError) as exc:
                logger.error(
                    "llm_provider_error", extra={"task": task, "error_type": type(exc).__name__}
                )
                raise ProviderError(details={"error_type": type(exc).__name__}) from exc
        raise InvalidModelOutputError()
