from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from ..models import RouteConfig
from .openai_compatible import OpenAICompatibleAdapter
from .responses import ResponsesAdapter


def _require_host(route: RouteConfig, *suffixes: str) -> None:
    host = (urlsplit(route.base_url).hostname or "").casefold()
    if not any(host == suffix or host.endswith("." + suffix) for suffix in suffixes):
        expected = ", ".join(suffixes)
        raise RuntimeError(f"{route.adapter} endpoint host must belong to {expected}")


class BedrockMantleAdapter(OpenAICompatibleAdapter):
    """AWS Bedrock Mantle's OpenAI-compatible chat-completions transport."""

    def preflight(self, route: RouteConfig) -> None:
        _require_host(route, "api.aws")
        path = urlsplit(route.base_url).path.rstrip("/")
        if not path.endswith("/chat/completions"):
            raise RuntimeError("Bedrock Mantle route must end in /chat/completions")
        if route.auth.header.casefold() != "authorization":
            raise RuntimeError("Bedrock Mantle API keys use the Authorization header")
        super().preflight(route)


class AzureModelInferenceAdapter(OpenAICompatibleAdapter):
    """Azure AI Foundry chat transport for services.ai.azure.com/openai.azure.com."""

    def preflight(self, route: RouteConfig) -> None:
        _require_host(route, "services.ai.azure.com", "openai.azure.com")
        path = urlsplit(route.base_url).path.rstrip("/")
        if not path.endswith("/chat/completions"):
            raise RuntimeError("Azure chat route must end in /chat/completions")
        if route.auth.header.casefold() not in {"api-key", "authorization"}:
            raise RuntimeError("Azure routes require api-key or Authorization authentication")
        super().preflight(route)


class BedrockMantleResponsesAdapter(ResponsesAdapter):
    def preflight(self, route: RouteConfig) -> None:
        _require_host(route, "api.aws")
        if not urlsplit(route.base_url).path.rstrip("/").endswith("/responses"):
            raise RuntimeError("Bedrock Mantle Responses route must end in /responses")
        super().preflight(route)


class AzureResponsesAdapter(ResponsesAdapter):
    def preflight(self, route: RouteConfig) -> None:
        _require_host(route, "services.ai.azure.com", "openai.azure.com")
        if not urlsplit(route.base_url).path.rstrip("/").endswith("/responses"):
            raise RuntimeError("Azure Responses route must end in /responses")
        if route.auth.header.casefold() not in {"api-key", "authorization"}:
            raise RuntimeError("Azure routes require api-key or Authorization authentication")
        super().preflight(route)


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """OpenRouter chat with an exact upstream provider and fallbacks disabled in the payload."""

    def preflight(self, route: RouteConfig) -> None:
        _require_host(route, "openrouter.ai")
        if not route.upstream_provider:
            raise RuntimeError("OpenRouter benchmark routes must pin one upstream provider")
        if urlsplit(route.base_url).path.rstrip("/") != "/api/v1/chat/completions":
            raise RuntimeError("OpenRouter chat route must use /api/v1/chat/completions")
        super().preflight(route)

    def headers(self, route: RouteConfig) -> dict[str, str]:
        headers = super().headers(route)
        headers["X-OpenRouter-Metadata"] = "enabled"
        return headers

    @staticmethod
    def _provider_aliases(value: str) -> frozenset[str]:
        canonical = re.sub(r"[^a-z0-9]", "", value.casefold().split("/", 1)[0])
        if canonical in {"google", "googlevertex"}:
            return frozenset({"google", "googlevertex"})
        return frozenset({canonical})

    def observe_provider_metadata(self, route: RouteConfig, data: dict[str, object]) -> bool:
        metadata = data.get("openrouter_metadata")
        if metadata is None:
            return False
        if not isinstance(metadata, dict):
            raise ValueError("OpenRouter metadata is not an object")
        endpoints = metadata.get("endpoints")
        available = endpoints.get("available") if isinstance(endpoints, dict) else None
        selected = [
            row for row in available or [] if isinstance(row, dict) and row.get("selected")
        ]
        if len(selected) != 1:
            raise ValueError("OpenRouter did not report exactly one selected endpoint")
        observed = [str(selected[0].get("provider") or "")]
        attempts = metadata.get("attempts")
        if isinstance(attempts, list):
            observed.extend(
                str(row.get("provider") or "") for row in attempts if isinstance(row, dict)
            )
        expected = self._provider_aliases(str(route.upstream_provider))
        if any(not (self._provider_aliases(value) & expected) for value in observed):
            raise ValueError("OpenRouter served a provider outside the hard pin")
        return True

    def requires_provider_metadata(self, route: RouteConfig) -> bool:
        return True


class VertexOpenAIAdapter(OpenAICompatibleAdapter):
    """Vertex AI's OpenAI-compatible endpoint with refreshable Google OAuth credentials."""

    _SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._credentials = None

    def _load_credentials(self, route: RouteConfig):  # type: ignore[no-untyped-def]
        try:
            import google.auth
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - exercised by installation preflight
            raise RuntimeError(
                "Vertex support requires the 'vertex' extra: pip install .[vertex]"
            ) from exc

        configured = os.environ.get(route.auth.env)
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    f"{route.auth.env} must name a readable service-account JSON file"
                )
            credentials = service_account.Credentials.from_service_account_file(
                str(path), scopes=self._SCOPES
            )
        else:
            credentials, _ = google.auth.default(scopes=self._SCOPES)
        return credentials

    def headers(self, route: RouteConfig) -> dict[str, str]:
        try:
            from google.auth.transport.requests import Request
        except ImportError as exc:  # pragma: no cover - exercised by installation preflight
            raise RuntimeError(
                "Vertex support requires the 'vertex' extra: pip install .[vertex]"
            ) from exc

        credentials = self._credentials or self._load_credentials(route)
        if not credentials.valid or not credentials.token:
            credentials.refresh(Request())
        self._credentials = credentials
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            **route.extra_headers,
        }
        return headers

    def preflight(self, route: RouteConfig) -> None:
        _require_host(route, "googleapis.com")
        path = urlsplit(route.base_url).path.rstrip("/")
        if not path.endswith("/chat/completions"):
            raise RuntimeError("Vertex OpenAI route must end in /chat/completions")
        super().preflight(route)
