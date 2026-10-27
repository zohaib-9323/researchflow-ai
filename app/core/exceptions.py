"""Domain exceptions.

Every exception carries a stable machine-readable `code`, an HTTP status and a
user-safe message. Messages must never contain secrets, file paths or stack traces.
"""

from __future__ import annotations


class ResearchFlowError(Exception):
    code = "internal_error"
    status_code = 500
    default_message = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, details: dict | None = None) -> None:
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(ResearchFlowError):
    code = "not_found"
    status_code = 404
    default_message = "The requested resource was not found."


class ValidationFailedError(ResearchFlowError):
    code = "validation_error"
    status_code = 422
    default_message = "The request is invalid."


class UnsupportedFileError(ValidationFailedError):
    code = "unsupported_file"
    status_code = 415
    default_message = "Only PDF and plain-text (.txt) files are supported."


class FileTooLargeError(ValidationFailedError):
    code = "file_too_large"
    status_code = 413


class DuplicateDocumentError(ResearchFlowError):
    code = "duplicate_document"
    status_code = 409
    default_message = "This document already exists in the collection."


class ExtractionError(ResearchFlowError):
    """The document could not be turned into searchable text."""

    code = "extraction_failed"
    status_code = 422
    default_message = "Text could not be extracted from the document."


class IndexingError(ResearchFlowError):
    code = "indexing_failed"
    status_code = 500
    default_message = "The document could not be indexed."


class EmbeddingUnavailableError(ResearchFlowError):
    code = "embeddings_unavailable"
    status_code = 503
    default_message = "The embedding model is not available in this environment."


class LLMNotConfiguredError(ResearchFlowError):
    code = "llm_not_configured"
    status_code = 503
    default_message = (
        "No Gemini API key is configured. Add GOOGLE_API_KEY_1 (and optional fallback "
        "keys GOOGLE_API_KEY_2..5) to .env.local and restart the API."
    )


class ProviderError(ResearchFlowError):
    """The LLM provider failed in a way that could not be recovered."""

    code = "provider_error"
    status_code = 502
    default_message = "The language model provider returned an error."


class ProviderUnavailableError(ProviderError):
    """All keys are exhausted / rate limited or the provider is down."""

    code = "provider_unavailable"
    status_code = 503
    default_message = (
        "The language model provider is temporarily unavailable or all API keys are "
        "rate limited. Please try again shortly."
    )


class InvalidModelOutputError(ProviderError):
    code = "invalid_model_output"
    status_code = 502
    default_message = "The language model returned output that could not be validated."


class AuthenticationRequiredError(ResearchFlowError):
    code = "authentication_required"
    status_code = 401
    default_message = "Sign in to use ResearchFlow AI."


class ForbiddenError(ResearchFlowError):
    code = "forbidden"
    status_code = 403
    default_message = "You don't have permission to do that."


class CreditsExhaustedError(ResearchFlowError):
    code = "credits_exhausted"
    status_code = 429
    default_message = "You have used all your free credits for this feature."
