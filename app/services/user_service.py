"""Signed-in users and their free credits.

Sign-in itself happens in the Streamlit UI (Google OpenID Connect via `st.login`). The UI
forwards the verified email to the API together with an internal token that only the UI
knows, so the API can trust the email without exposing its own login flow. With
`AUTH_ENABLED=false` (local development) there are no users and no limits.

Credits are reserved before the work starts and refunded if it fails, so a provider
outage doesn't cost the user a credit.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy.orm import Session

from app.config import Settings
from app.core.exceptions import (
    AuthenticationRequiredError,
    CreditsExhaustedError,
    ForbiddenError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.persistence.database import Database
from app.persistence.models import UserAccount

logger = get_logger(__name__)

# Identity used by in-process maintenance (demo seeding); always an administrator.
SYSTEM_EMAIL = "system@researchflow.internal"
CreditKind = Literal["question", "agent_run"]
_USED_FIELD = {"question": "questions_used", "agent_run": "agent_runs_used"}
_LABEL = {"question": "questions", "agent_run": "deep research runs"}


@dataclass(frozen=True)
class CurrentUser:
    email: str
    is_admin: bool
    name: str | None = None


class UserService:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database

    @property
    def enabled(self) -> bool:
        return self.settings.auth_enabled

    def authenticate(
        self, token: str | None, email: str | None, name: str | None = None
    ) -> CurrentUser | None:
        """The caller, or None when sign-in is disabled. Raises when it is required."""
        if not self.enabled:
            return None
        expected = self.settings.internal_api_token
        if (
            expected is None
            or not token
            or not hmac.compare_digest(token.encode(), expected.get_secret_value().encode())
        ):
            raise AuthenticationRequiredError("This request is not authorized.")
        email = (email or "").strip().lower()
        if "@" not in email or len(email) > 320:
            raise AuthenticationRequiredError()
        user = CurrentUser(
            email=email,
            is_admin=email in self.settings.admin_email_set or email == SYSTEM_EMAIL,
            name=(name or "").strip()[:200] or None,
        )
        self._touch(user)
        return user

    def limit(self, kind: CreditKind) -> int:
        return self.settings.free_questions if kind == "question" else self.settings.free_agent_runs

    def reserve(self, user: CurrentUser | None, kind: CreditKind) -> None:
        """Use one credit, or raise if none are left. Admins and anonymous local use are
        unlimited."""
        if user is None or user.is_admin:
            return
        field = _USED_FIELD[kind]
        with self.database.session() as session:
            account = session.get(UserAccount, user.email, with_for_update=True)
            if account is None:
                account = self._create(session, user)
            used = getattr(account, field)
            if used >= self.limit(kind):
                raise CreditsExhaustedError(
                    f"You have used your {self.limit(kind)} free {_LABEL[kind]}."
                )
            setattr(account, field, used + 1)
        logger.info("credit_used", extra={"kind": kind})

    def refund(self, user: CurrentUser | None, kind: CreditKind) -> None:
        if user is None or user.is_admin:
            return
        field = _USED_FIELD[kind]
        with self.database.session() as session:
            account = session.get(UserAccount, user.email)
            if account is not None and getattr(account, field) > 0:
                setattr(account, field, getattr(account, field) - 1)

    def status(self, user: CurrentUser | None) -> dict:
        if user is None:
            return {"auth_enabled": False, "email": None, "is_admin": True, "credits": None}
        with self.database.session() as session:
            account = session.get(UserAccount, user.email)
            used = {
                kind: getattr(account, field) if account else 0
                for kind, field in _USED_FIELD.items()
            }
        credits = {
            kind: {
                "limit": None if user.is_admin else self.limit(kind),  # type: ignore[arg-type]
                "used": used[kind],
                "remaining": None if user.is_admin else max(0, self.limit(kind) - used[kind]),  # type: ignore[arg-type]
            }
            for kind in _USED_FIELD
        }
        return {
            "auth_enabled": True,
            "email": user.email,
            "name": user.name,
            "is_admin": user.is_admin,
            "credits": credits,
        }

    # -- access checks --------------------------------------------------------------------
    @staticmethod
    def check_owner(user: CurrentUser | None, owner_email: str | None) -> None:
        """Hide other users' conversations and runs (404, so ids can't be probed)."""
        if user is None or user.is_admin:
            return
        if owner_email != user.email:
            raise NotFoundError()

    @staticmethod
    def require_admin(user: CurrentUser | None) -> None:
        if user is not None and not user.is_admin:
            raise ForbiddenError("Only the administrator can change the document library.")

    @staticmethod
    def owner_filter(user: CurrentUser | None) -> str | None:
        """Email to filter listings by, or None to list everything."""
        return None if user is None or user.is_admin else user.email

    # -- internals ------------------------------------------------------------------------
    def _touch(self, user: CurrentUser) -> None:
        with self.database.session() as session:
            account = session.get(UserAccount, user.email)
            if account is None:
                self._create(session, user)
            else:
                account.last_seen_at = datetime.now(UTC)
                if user.name:
                    account.name = user.name

    @staticmethod
    def _create(session: Session, user: CurrentUser) -> UserAccount:
        account = UserAccount(email=user.email, name=user.name)
        session.add(account)
        session.flush()
        return account
