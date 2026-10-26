from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.exceptions import ModelAPIError
from langchain_google_genai.chat_models import (
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)

from app.core.exceptions import InvalidModelOutputError, ProviderError, ProviderUnavailableError
from app.llm.key_pool import FailureKind, KeyPool
from app.llm.models import LLMGateway, classify_provider_error
from app.llm.prompts import CLASSIFY_PROMPT, UNTRUSTED_DATA_RULES
from app.llm.structured_outputs import QuestionClassification
from tests.conftest import make_settings
from tests.fakes import FakeStructuredChatModel

RATE_LIMIT_MSG = (
    "Error calling model 'gemini' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
    "'retryDelay': '41s'}"
)
INPUTS = {
    "question": "What did the pilot cost?",
    "history": "",
    "security_rules": UNTRUSTED_DATA_RULES,
}
GOOD = QuestionClassification(
    question_type="document_question",
    needs_retrieval=True,
    standalone_question="What did the pilot cost?",
    search_query="pilot cost",
    reason="r",
)


def gateway(tmp_path: Path, models: dict[str, FakeStructuredChatModel], **overrides: object):
    settings = make_settings(tmp_path, **overrides)
    pool = KeyPool(list(models), clock=lambda: 0.0)
    return LLMGateway(settings, pool, model_factory=lambda key, model: models[key]), pool


def run(gw: LLMGateway) -> QuestionClassification:
    return gw.structured(CLASSIFY_PROMPT, QuestionClassification, INPUTS, task="classify")


def test_structured_output_is_validated(tmp_path: Path) -> None:
    gw, _ = gateway(tmp_path, {"k1": FakeStructuredChatModel(responses=[GOOD])})
    assert run(gw).search_query == "pilot cost"


def test_rate_limit_fails_over_to_next_key_and_sticks(tmp_path: Path) -> None:
    k1 = FakeStructuredChatModel(responses=[GoogleRateLimitError(RATE_LIMIT_MSG)])
    k2 = FakeStructuredChatModel(responses=[GOOD])
    gw, pool = gateway(tmp_path, {"k1": k1, "k2": k2})
    assert run(gw) == GOOD
    assert pool.active_slot == 2
    assert pool.status()["cooling_down"][0]["entry"] == "key 1"
    run(gw)
    assert (k1.calls, k2.calls) == (1, 2)  # second call went straight to k2


def test_invalid_output_is_retried_then_rejected(tmp_path: Path) -> None:
    bad = {"raw": None, "parsed": None, "parsing_error": ValueError("not json")}
    model = FakeStructuredChatModel(responses=[bad, GOOD])
    gw, _ = gateway(tmp_path, {"k1": model}, llm_invalid_output_retries=1)
    assert run(gw) == GOOD

    always_bad = FakeStructuredChatModel(responses=[bad])
    gw, _ = gateway(tmp_path, {"k1": always_bad}, llm_invalid_output_retries=1)
    with pytest.raises(InvalidModelOutputError):
        run(gw)
    assert always_bad.calls == 2


def test_schema_violations_count_as_invalid_output(tmp_path: Path) -> None:
    wrong = {"raw": None, "parsed": {"question_type": "nonsense"}, "parsing_error": None}
    gw, _ = gateway(
        tmp_path, {"k1": FakeStructuredChatModel(responses=[wrong])}, llm_invalid_output_retries=0
    )
    with pytest.raises(InvalidModelOutputError):
        run(gw)


def test_all_keys_rate_limited(tmp_path: Path) -> None:
    models = {
        k: FakeStructuredChatModel(responses=[GoogleRateLimitError(RATE_LIMIT_MSG)])
        for k in ("k1", "k2")
    }
    gw, _ = gateway(tmp_path, models, llm_max_wait_seconds=0)
    with pytest.raises(ProviderUnavailableError):
        run(gw)


def test_model_not_found_is_a_provider_error_not_failover(tmp_path: Path) -> None:
    k1 = FakeStructuredChatModel(responses=[GoogleModelNotFoundError("404 model missing")])
    k2 = FakeStructuredChatModel(responses=[GOOD])
    gw, _ = gateway(tmp_path, {"k1": k1, "k2": k2})
    with pytest.raises(ProviderError, match="CHAT_MODEL"):
        run(gw)
    assert k2.calls == 0


def test_error_classification(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, key_cooldown_seconds=60, key_daily_quota_cooldown_seconds=3600
    )
    limited = classify_provider_error(GoogleRateLimitError(RATE_LIMIT_MSG), settings)
    assert limited is not None
    assert (limited.kind, limited.cooldown_seconds) == (FailureKind.RATE_LIMITED, 41.0)

    daily = classify_provider_error(
        GoogleRateLimitError("429 quotaId: GenerateRequestsPerDayPerProjectPerModel"), settings
    )
    assert daily is not None
    assert daily.cooldown_seconds == 3600

    invalid = classify_provider_error(
        GoogleInvalidRequestError("400 API_KEY_INVALID API key not valid"), settings
    )
    assert invalid is not None
    assert invalid.kind is FailureKind.INVALID_KEY
    auth = classify_provider_error(GoogleAuthenticationError("401"), settings)
    assert auth is not None
    assert auth.kind is FailureKind.INVALID_KEY

    transient = classify_provider_error(ModelAPIError("503 overloaded"), settings)
    assert transient is not None
    assert transient.kind is FailureKind.TRANSIENT

    assert classify_provider_error(GoogleInvalidRequestError("400 bad schema"), settings) is None
    assert classify_provider_error(ValueError("bug"), settings) is None


def test_records_which_model_answered(tmp_path: Path) -> None:
    from app.llm.models import models_used_var

    settings = make_settings(tmp_path)
    limited = FakeStructuredChatModel(responses=[GoogleRateLimitError(RATE_LIMIT_MSG)])
    fallback = FakeStructuredChatModel(responses=[GOOD])
    pool = KeyPool(["k1"], ["flash", "lite"], clock=lambda: 0.0)
    gw = LLMGateway(
        settings, pool, model_factory=lambda key, model: limited if model == "flash" else fallback
    )
    used: list[str] = []
    token = models_used_var.set(used)
    try:
        assert run(gw) == GOOD
    finally:
        models_used_var.reset(token)
    assert used == ["lite"]


def test_redact_secrets_removes_google_keys() -> None:
    from app.llm.models import redact_secrets

    fake_key = "AIza" + "x" * 35
    assert fake_key not in redact_secrets(f"403 for key {fake_key}: denied")
    assert redact_secrets("plain message") == "plain message"
