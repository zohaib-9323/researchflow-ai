from __future__ import annotations

from fastapi import Depends, Request

from app.container import Container
from app.services.user_service import CurrentUser

TOKEN_HEADER = "X-Internal-Token"  # noqa: S105 (header name, not a secret)
EMAIL_HEADER = "X-User-Email"
NAME_HEADER = "X-User-Name"


def get_container(request: Request) -> Container:
    return request.app.state.container


def current_user(
    request: Request, container: Container = Depends(get_container)
) -> CurrentUser | None:
    """The signed-in user forwarded by the UI (None when sign-in is disabled)."""
    return container.users.authenticate(
        request.headers.get(TOKEN_HEADER),
        request.headers.get(EMAIL_HEADER),
        request.headers.get(NAME_HEADER),
    )
