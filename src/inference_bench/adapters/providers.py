from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ..models import RouteConfig
from .openai_compatible import OpenAICompatibleAdapter
from .responses import ResponsesAdapter

_ALIBABA_WORKSPACE_HOST = re.compile(
    r"^(?P<workspace>[a-z0-9][a-z0-9-]*)\."
    r"(?P<region>cn-beijing|ap-southeast-1|ap-northeast-1|eu-central-1|"
    r"cn-hongkong|us-east-1)\.maas\.aliyuncs\.com$"
)
_ALIBABA_DASHSCOPE_REGIONS = {
    "dashscope.aliyuncs.com": "cn-beijing",
    "dashscope-intl.aliyuncs.com": "ap-southeast-1",
    "dashscope-us.aliyuncs.com": "us-east-1",
    "cn-hongkong.dashscope.aliyuncs.com": "cn-hongkong",
}
_ALIBABA_ADAPTIVE_429_CODES = frozenset(
    code.casefold()
    for code in (
        "Throttling.RateQuota",
        "LimitRequests",
        "limit_requests",
        "Throttling.AllocationQuota",
        "insufficient_quota",
        "Throttling.BurstRate",
        "limit_burst_rate",
    )
)
_ALIBABA_ACCOUNT_429_CODES = frozenset(
    code.casefold()
    for code in (
        "CommodityNotPurchased",
        "PrepaidBillOverdue",
        "PostpaidBillOverdue",
    )
)


def _require_host(route: RouteConfig, *suffixes: str) -> None:
    host = (urlsplit(route.base_url).hostname or "").casefold()
    if not any(host == suffix or host.endswith("." + suffix) for suffix in suffixes):
        expected = ", ".join(suffixes)
        raise RuntimeError(f"{route.adapter} endpoint host must belong to {expected}")


def _preflight_alibaba_payg(route: RouteConfig, expected_path: str) -> None:
    parsed = urlsplit(route.base_url)
    if route.provider != "alibaba-model-studio":
        raise RuntimeError(
            "Alibaba Model Studio adapters require provider=alibaba-model-studio"
        )
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "Alibaba Model Studio pay-as-you-go routes require a canonical HTTPS endpoint "
            "without credentials, a custom port, query parameters, or a fragment"
        )
    host = (parsed.hostname or "").casefold()
    workspace_match = _ALIBABA_WORKSPACE_HOST.fullmatch(host)
    if workspace_match is not None:
        workspace = workspace_match.group("workspace")
        if workspace in {"token-plan", "trial"} or workspace.startswith("replace-with-"):
            workspace_match = None
    observed_region = (
        workspace_match.group("region")
        if workspace_match is not None
        else _ALIBABA_DASHSCOPE_REGIONS.get(host)
    )
    if observed_region is None:
        raise RuntimeError(
            "Alibaba Model Studio route must use an official pay-as-you-go workspace or "
            "DashScope domain"
        )
    if route.billing_channel != "pay_as_you_go":
        raise RuntimeError("Alibaba automated benchmarking requires billing_channel=pay_as_you_go")
    if route.region != observed_region:
        raise RuntimeError(
            "Alibaba Model Studio API keys, model catalogs, and endpoints are region-bound"
        )
    if parsed.path.rstrip("/") != expected_path:
        raise RuntimeError(f"Alibaba Model Studio route must use {expected_path}")
    if route.auth.header.casefold() != "authorization" or route.auth.prefix != "Bearer ":
        raise RuntimeError(
            "Alibaba Model Studio OpenAI-compatible routes require Authorization: Bearer"
        )


def _alibaba_error_code(raw: bytes) -> str | None:
    if len(raw) > 1_000_000:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    values = [payload.get("code")]
    nested = payload.get("error")
    if isinstance(nested, dict):
        values.extend((nested.get("code"), nested.get("type")))
    return next((value for value in values if isinstance(value, str) and value), None)


class _AlibabaErrorClassificationMixin:
    def _classify_http_error(
        self, response: httpx.Response, raw: bytes
    ) -> tuple[str, str]:
        code = _alibaba_error_code(raw)
        normalized = code.casefold() if code is not None else None
        if response.status_code == 429:
            if normalized in _ALIBABA_ACCOUNT_429_CODES:
                return "client_error", "provider_billing_or_entitlement"
            if normalized in _ALIBABA_ADAPTIVE_429_CODES:
                return "rate_limited", "provider_rate_limit"
        if response.status_code in {400, 401, 403, 404}:
            return "client_error", "provider_route_fatal"
        return super()._classify_http_error(response, raw)  # type: ignore[misc]


class AlibabaModelStudioAdapter(_AlibabaErrorClassificationMixin, OpenAICompatibleAdapter):
    """Alibaba Model Studio pay-as-you-go OpenAI-compatible chat transport."""

    def preflight(self, route: RouteConfig) -> None:
        _preflight_alibaba_payg(route, "/compatible-mode/v1/chat/completions")
        super().preflight(route)


class AlibabaModelStudioResponsesAdapter(_AlibabaErrorClassificationMixin, ResponsesAdapter):
    """Alibaba Model Studio pay-as-you-go OpenAI-compatible Responses transport."""

    def preflight(self, route: RouteConfig) -> None:
        _preflight_alibaba_payg(route, "/compatible-mode/v1/responses")
        super().preflight(route)


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
        selected = [row for row in available or [] if isinstance(row, dict) and row.get("selected")]
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
