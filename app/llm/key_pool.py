"""Rotating pool of Gemini credentials (API key + model) with sticky failover.

Free-tier quotas are counted per Google Cloud project *and per model*, so the pool is
made of (key, model) entries ordered model-first: every key is tried on the primary
model before any key falls back to the next model.

Behaviour:
* One entry is *active*. Every request starts on the active entry — not on key #1.
* When the active key is exhausted (HTTP 429 / quota) or rejected (invalid, revoked,
  permission denied), it is benched for a cooldown and the pool advances to the next
  available key. The failed request is retried on that key immediately, and all later
  requests continue from it ("pick up right from there").
* Cooldowns use the provider's `retryDelay` when present, a longer bench for daily
  quota exhaustion, and a very long bench for invalid keys.
* Transient provider failures (5xx, overload, timeouts) are not the key's fault: the
  request is retried on the next key but the active key does not change.
* The active key and cooldowns are persisted (as key *fingerprints*, never the keys)
  so a restart resumes where the pool left off.

Keys are only ever referred to by 1-based key number in logs and API output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

from app.core.exceptions import LLMNotConfiguredError, ProviderUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

INVALID_KEY_COOLDOWN_SECONDS = 24 * 3600.0
_RETRY_DELAY = re.compile(r"retry(?:Delay|_delay)['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I)
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I)


class FailureKind(Enum):
    RATE_LIMITED = "rate_limited"
    INVALID_KEY = "invalid_key"
    TRANSIENT = "transient"


@dataclass
class KeyFailure:
    kind: FailureKind
    cooldown_seconds: float = 0.0


@dataclass(frozen=True)
class PoolEntry:
    key: str
    model: str
    key_number: int  # 1-based position of the key in the configuration

    @property
    def label(self) -> str:
        return f"key {self.key_number} / {self.model}" if self.model else f"key {self.key_number}"


def fingerprint(key: str, model: str = "") -> str:
    return hashlib.sha256(f"{model}\x00{key}".encode()).hexdigest()[:16]


def parse_retry_delay(message: str) -> float | None:
    for pattern in (_RETRY_DELAY, _RETRY_IN):
        if match := pattern.search(message):
            return float(match.group(1))
    return None


class KeyPool:
    def __init__(
        self,
        keys: list[str],
        models: list[str] | None = None,
        *,
        state_path: Path | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        max_wait_seconds: float = 0.0,
    ) -> None:
        self._key_count = len(keys)
        self._entries = [
            PoolEntry(key=key, model=model, key_number=number)
            for model in (models or [""])
            for number, key in enumerate(keys, start=1)
        ]
        self._fingerprints = [fingerprint(e.key, e.model) for e in self._entries]
        self._cooldown_until: list[float] = [0.0] * len(self._entries)
        self._active = 0
        self._state_path = state_path
        self._clock = clock
        self._sleep = sleep
        # When every key is cooling down, wait up to this long for the first to recover
        # (useful for per-minute quotas) instead of failing immediately.
        self._max_wait = max_wait_seconds
        self._lock = threading.Lock()
        self._load_state()

    # -- introspection -----------------------------------------------------------------
    def __len__(self) -> int:
        return self._key_count

    @property
    def active_entry(self) -> PoolEntry | None:
        return self._entries[self._active] if self._entries else None

    @property
    def active_slot(self) -> int:
        """1-based number of the active key (0 when no keys)."""
        entry = self.active_entry
        return entry.key_number if entry else 0

    def status(self) -> dict:
        now = self._clock()
        with self._lock:
            benched = [
                {"entry": self._entries[i].label, "available_in_seconds": round(until - now)}
                for i, until in enumerate(self._cooldown_until)
                if until > now
            ]
            active = self.active_entry
            return {
                "total_keys": self._key_count,
                "models": list(dict.fromkeys(e.model for e in self._entries if e.model)),
                "active_slot": self.active_slot,
                "active_model": active.model if active else None,
                "available_entries": len(self._entries) - len(benched),
                "total_entries": len(self._entries),
                "cooling_down": benched,
            }

    # -- core ------------------------------------------------------------------------------
    def execute(
        self,
        operation: Callable[[PoolEntry], T],
        classify: Callable[[BaseException], KeyFailure | None],
        *,
        max_transient_failures: int = 3,
    ) -> T:
        """Run `operation(entry)`, failing over across pool entries as needed.

        `classify` maps an exception to a KeyFailure, or None for errors that are not
        key/provider related (those propagate immediately).
        """
        if not self._entries:
            raise LLMNotConfiguredError()
        tried: set[int] = set()
        last_failure: KeyFailure | None = None
        transient_failures = 0
        waited = 0.0
        while transient_failures < max_transient_failures:
            index = self._next_candidate(tried)
            if index is None:
                delay = self._seconds_until_recovery()
                if delay is None or waited + delay > self._max_wait:
                    break
                logger.info("llm_waiting_for_key", extra={"wait_seconds": round(delay, 1)})
                self._sleep(delay)
                waited += delay
                # Keys whose cooldown has expired may be tried again.
                tried = {i for i in tried if self._cooldown_until[i] > self._clock()}
                continue
            tried.add(index)
            try:
                result = operation(self._entries[index])
            except Exception as exc:
                failure = classify(exc)
                if failure is None:
                    raise
                last_failure = failure
                if failure.kind is FailureKind.TRANSIENT:
                    transient_failures += 1
                self._record_failure(index, failure)
                continue
            self._record_success(index)
            return result
        message = (
            "All Gemini API keys (and fallback models) are currently rate limited or "
            "exhausted. Please retry later, add a key from a different Google Cloud "
            "project (keys in the same project share one quota) or set "
            "CHAT_FALLBACK_MODELS."
            if last_failure is None or last_failure.kind is not FailureKind.TRANSIENT
            else None
        )
        raise ProviderUnavailableError(message, details=self.status())

    def _seconds_until_recovery(self) -> float | None:
        """Seconds until the first rate-limited key recovers (None if none will soon)."""
        now = self._clock()
        with self._lock:
            pending = [until - now for until in self._cooldown_until if until > now]
        if not pending:
            return None
        delay = min(pending)
        return None if delay >= INVALID_KEY_COOLDOWN_SECONDS / 2 else delay + 0.5

    def _next_candidate(self, tried: set[int]) -> int | None:
        """Active key first, then the following keys in order, skipping benched ones."""
        now = self._clock()
        with self._lock:
            count = len(self._entries)
            for offset in range(count):
                index = (self._active + offset) % count
                if index not in tried and self._cooldown_until[index] <= now:
                    return index
        return None

    def _record_failure(self, index: int, failure: KeyFailure) -> None:
        with self._lock:
            if failure.kind is not FailureKind.TRANSIENT:
                until = self._clock() + failure.cooldown_seconds
                if failure.kind is FailureKind.INVALID_KEY:
                    # A rejected key is unusable for every model.
                    bad_key = self._entries[index].key
                    for i, entry in enumerate(self._entries):
                        if entry.key == bad_key:
                            self._cooldown_until[i] = until
                self._cooldown_until[index] = until
                if index == self._active:
                    self._active = self._first_available_after(index)
            self._save_state_locked()
        logger.warning(
            "llm_key_failover",
            extra={
                "failed": self._entries[index].label,
                "failure": failure.kind.value,
                "cooldown_seconds": round(failure.cooldown_seconds),
                "now_active": self._entries[self._active].label,
            },
        )

    def _first_available_after(self, index: int) -> int:
        now = self._clock()
        count = len(self._entries)
        for offset in range(1, count + 1):
            candidate = (index + offset) % count
            if self._cooldown_until[candidate] <= now:
                return candidate
        # Everything is benched: the key that recovers first becomes active.
        return min(range(count), key=lambda i: self._cooldown_until[i])

    def _record_success(self, index: int) -> None:
        with self._lock:
            changed = index != self._active
            self._active = index
            if changed:
                self._save_state_locked()

    # -- persistence -------------------------------------------------------------------------
    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
        except (OSError, ValueError):
            logger.warning("key_pool_state_unreadable")
            return
        active = state.get("active")
        if active in self._fingerprints:
            self._active = self._fingerprints.index(active)
        for fp, until in (state.get("cooldowns") or {}).items():
            if fp in self._fingerprints and isinstance(until, int | float):
                self._cooldown_until[self._fingerprints.index(fp)] = float(until)

    def _save_state_locked(self) -> None:
        if self._state_path is None:
            return
        now = self._clock()
        state = {
            "active": self._fingerprints[self._active],
            "cooldowns": {
                fp: until
                for fp, until in zip(self._fingerprints, self._cooldown_until, strict=True)
                if until > now
            },
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, self._state_path)
        except OSError:
            logger.warning("key_pool_state_not_saved")
