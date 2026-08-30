from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

_CLOUD_PLATFORM_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


class _AuthConfig(Protocol):
    env: str


class _GoogleRoute(Protocol):
    auth: _AuthConfig
    extra_headers: dict[str, str]


class GoogleOAuthBearer:
    """Refreshable Google OAuth headers shared by every native Vertex transport.

    Credential loading is deliberately lazy. Planning and report generation therefore remain
    credential-free, while a live adapter refreshes before it crosses the durable request-claim
    boundary. Tokens are returned only as transient request headers and are never serialized.
    """

    def __init__(
        self,
        *,
        credentials: Any | None = None,
        credential_loader: Callable[[_GoogleRoute], Any] | None = None,
        auth_request_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._injected_credentials = credentials
        self._credential_loader = credential_loader or self._load_credentials
        self._auth_request_factory = auth_request_factory
        self._credentials_by_env: dict[str, Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _load_credentials(route: _GoogleRoute) -> Any:
        try:
            import google.auth
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - installation preflight
            raise RuntimeError("Vertex support requires google-auth") from exc
        configured = os.environ.get(route.auth.env)
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    f"{route.auth.env} must name a readable service-account JSON file"
                )
            return service_account.Credentials.from_service_account_file(
                str(path), scopes=_CLOUD_PLATFORM_SCOPES
            )
        credentials, _ = google.auth.default(scopes=_CLOUD_PLATFORM_SCOPES)
        return credentials

    def _credentials(self, route: _GoogleRoute) -> Any:
        if self._injected_credentials is not None:
            return self._injected_credentials
        if route.auth.env not in self._credentials_by_env:
            self._credentials_by_env[route.auth.env] = self._credential_loader(route)
        return self._credentials_by_env[route.auth.env]

    def _auth_request(self) -> Any:
        if self._auth_request_factory is not None:
            return self._auth_request_factory()
        try:
            from google.auth.transport.requests import Request
        except ImportError as exc:  # pragma: no cover - installation preflight
            raise RuntimeError("Vertex support requires google-auth") from exc
        return Request()

    def headers(self, route: _GoogleRoute, *, accept: str) -> dict[str, str]:
        credentials = self._credentials(route)
        token = getattr(credentials, "token", None)
        if not bool(getattr(credentials, "valid", False)) or not token:
            with self._lock:
                token = getattr(credentials, "token", None)
                if not bool(getattr(credentials, "valid", False)) or not token:
                    credentials.refresh(self._auth_request())
                    token = getattr(credentials, "token", None)
        if not isinstance(token, str) or not token:
            raise RuntimeError("Google OAuth refresh did not produce an access token")
        headers = {
            **route.extra_headers,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            "Accept": accept,
        }
        if any(
            any(character in name or character in value for character in "\r\n\0")
            for name, value in headers.items()
        ):
            raise RuntimeError("constructed Google OAuth headers contain control characters")
        try:
            httpx.Headers(headers)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("constructed Google OAuth headers are invalid") from exc
        return headers


__all__ = ["GoogleOAuthBearer"]
