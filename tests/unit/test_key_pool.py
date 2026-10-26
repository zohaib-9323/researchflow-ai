from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import LLMNotConfiguredError, ProviderUnavailableError
from app.llm.key_pool import FailureKind, KeyFailure, KeyPool, PoolEntry, parse_retry_delay


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RateLimitedError(Exception):
    pass


class BadKeyError(Exception):
    pass


class OverloadedError(Exception):
    pass


def classify(exc: BaseException) -> KeyFailure | None:
    if isinstance(exc, RateLimitedError):
        return KeyFailure(FailureKind.RATE_LIMITED, 60)
    if isinstance(exc, BadKeyError):
        return KeyFailure(FailureKind.INVALID_KEY, 86400)
    if isinstance(exc, OverloadedError):
        return KeyFailure(FailureKind.TRANSIENT)
    return None


def make_operation(failing: dict[str, type[Exception]], log: list[str]):
    def operation(entry: PoolEntry) -> str:
        key = entry.key
        log.append(key)
        if key in failing:
            raise failing[key]("boom")
        return f"ok:{key}"

    return operation


def test_sticky_failover_continues_from_next_key() -> None:
    clock = Clock()
    pool = KeyPool(["k1", "k2", "k3"], clock=clock)
    log: list[str] = []
    failing: dict[str, type[Exception]] = {"k1": RateLimitedError}

    assert pool.execute(make_operation(failing, log), classify) == "ok:k2"
    assert log == ["k1", "k2"]
    assert pool.active_slot == 2

    # The next request starts on k2 directly, it does not retry k1 first.
    log.clear()
    assert pool.execute(make_operation(failing, log), classify) == "ok:k2"
    assert log == ["k2"]

    # k2 now exhausted -> continue to k3, still skipping benched k1.
    log.clear()
    failing["k2"] = RateLimitedError
    assert pool.execute(make_operation(failing, log), classify) == "ok:k3"
    assert log == ["k2", "k3"]
    assert pool.active_slot == 3


def test_wraps_around_after_cooldown_expires() -> None:
    clock = Clock()
    pool = KeyPool(["k1", "k2"], clock=clock)
    log: list[str] = []
    pool.execute(make_operation({"k1": RateLimitedError}, log), classify)  # k1 benched 60s
    failing = {"k2": RateLimitedError}
    clock.now += 61  # k1 recovered
    log.clear()
    assert pool.execute(make_operation(failing, log), classify) == "ok:k1"
    assert log == ["k2", "k1"]
    assert pool.active_slot == 1


def test_all_keys_exhausted_raises_provider_unavailable() -> None:
    pool = KeyPool(["k1", "k2"], clock=Clock())
    log: list[str] = []
    with pytest.raises(ProviderUnavailableError) as info:
        pool.execute(
            make_operation({"k1": RateLimitedError, "k2": RateLimitedError}, log), classify
        )
    assert log == ["k1", "k2"]
    assert info.value.details["available_entries"] == 0
    assert "k1" not in str(info.value.details)  # keys never leak into errors


def test_waits_for_short_cooldown_when_configured() -> None:
    clock = Clock()
    pool = KeyPool(["k1"], clock=clock, sleep=clock.sleep, max_wait_seconds=90)
    calls = {"n": 0}

    def operation(entry: PoolEntry) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitedError()
        return "recovered"

    assert pool.execute(operation, classify) == "recovered"
    assert clock.now >= 1060  # waited for the 60 s cooldown


def test_does_not_wait_beyond_budget() -> None:
    clock = Clock()
    pool = KeyPool(["k1"], clock=clock, sleep=clock.sleep, max_wait_seconds=10)
    with pytest.raises(ProviderUnavailableError):
        pool.execute(make_operation({"k1": RateLimitedError}, []), classify)
    assert clock.now == 1000.0


def test_invalid_key_is_benched_long_term() -> None:
    clock = Clock()
    pool = KeyPool(["bad", "good"], clock=clock)
    log: list[str] = []
    pool.execute(make_operation({"bad": BadKeyError}, log), classify)
    clock.now += 3600
    log.clear()
    pool.execute(make_operation({}, log), classify)
    assert log == ["good"]


def test_transient_errors_fail_over_without_moving_active_key() -> None:
    pool = KeyPool(["k1", "k2", "k3"], clock=Clock())
    log: list[str] = []
    assert pool.execute(make_operation({"k1": OverloadedError}, log), classify) == "ok:k2"
    assert pool.status()["cooling_down"] == []  # overload is not the key's fault


def test_transient_failures_are_bounded() -> None:
    pool = KeyPool([f"k{i}" for i in range(6)], clock=Clock())
    log: list[str] = []
    with pytest.raises(ProviderUnavailableError):
        pool.execute(
            make_operation({f"k{i}": OverloadedError for i in range(6)}, log),
            classify,
            max_transient_failures=3,
        )
    assert len(log) == 3


def test_unclassified_errors_propagate_immediately() -> None:
    pool = KeyPool(["k1", "k2"], clock=Clock())
    with pytest.raises(ValueError, match="bug"):
        pool.execute(lambda entry: (_ for _ in ()).throw(ValueError("bug")), classify)


def test_no_keys_means_not_configured() -> None:
    with pytest.raises(LLMNotConfiguredError):
        KeyPool([]).execute(lambda entry: "x", classify)


def test_state_persists_across_restarts_without_storing_keys(tmp_path: Path) -> None:
    state = tmp_path / "pool.json"
    clock = Clock()
    pool = KeyPool(["secret-a", "secret-b", "secret-c"], state_path=state, clock=clock)
    pool.execute(make_operation({"secret-a": RateLimitedError}, []), classify)
    assert "secret" not in state.read_text()

    restarted = KeyPool(["secret-a", "secret-b", "secret-c"], state_path=state, clock=clock)
    log: list[str] = []
    restarted.execute(make_operation({}, log), classify)
    assert log == ["secret-b"]  # resumed on the active key; k1 still benched


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("... 'retryDelay': '37s' ...", 37.0),
        ("Please retry in 12.5s.", 12.5),
        ("no hint", None),
    ],
)
def test_parse_retry_delay(message: str, expected: float | None) -> None:
    assert parse_retry_delay(message) == expected


def test_model_fallback_after_all_keys_exhausted_on_primary_model() -> None:
    pool = KeyPool(["k1", "k2"], ["flash", "lite"], clock=Clock())
    used: list[tuple[str, str]] = []

    def operation(entry: PoolEntry) -> str:
        used.append((entry.key, entry.model))
        if entry.model == "flash":
            raise RateLimitedError()
        return entry.model

    assert pool.execute(operation, classify) == "lite"
    # every key is tried on the primary model before falling back
    assert used == [("k1", "flash"), ("k2", "flash"), ("k1", "lite")]
    status = pool.status()
    assert (status["active_slot"], status["active_model"]) == (1, "lite")
    assert status["models"] == ["flash", "lite"]
    # sticky: the next call goes straight to the fallback entry
    used.clear()
    pool.execute(operation, classify)
    assert used == [("k1", "lite")]


def test_invalid_key_is_benched_for_every_model() -> None:
    pool = KeyPool(["bad", "good"], ["flash", "lite"], clock=Clock())
    used: list[tuple[str, str]] = []

    def operation(entry: PoolEntry) -> str:
        used.append((entry.key, entry.model))
        if entry.key == "bad":
            raise BadKeyError()
        if entry.model == "flash":
            raise RateLimitedError()
        return "ok"

    assert pool.execute(operation, classify) == "ok"
    assert ("bad", "lite") not in used
    assert used == [("bad", "flash"), ("good", "flash"), ("good", "lite")]
