from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Protocol

from ..models import InferenceResult, RequestSpec, RouteConfig, sha256_json
from ..payload import MaterializedPayload


class AdapterUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """Exact request materialization handed unchanged across the durable claim boundary."""

    payload: MaterializedPayload
    headers: dict[str, str] = field(default_factory=dict)


class Adapter(Protocol):
    def preflight(self, route: RouteConfig) -> None: ...

    def prepare(self, route: RouteConfig, request: RequestSpec) -> PreparedRequest: ...

    async def send_prepared(
        self, route: RouteConfig, request: RequestSpec, prepared: PreparedRequest
    ) -> InferenceResult: ...

    async def close(self) -> None: ...


class FailClosedAdapter:
    def __init__(self, name: str) -> None:
        self.name = name

    def preflight(self, route: RouteConfig) -> None:
        raise AdapterUnavailable(
            f"adapter {self.name!r} is an honest placeholder for {route.provider}"
        )

    def prepare(self, route: RouteConfig, request: RequestSpec) -> PreparedRequest:
        raise AdapterUnavailable(
            f"adapter {self.name!r} is an honest placeholder; implement and contract-test "
            f"the exact {route.provider}/{route.api_family} materialization before live use"
        )

    async def send_prepared(
        self, route: RouteConfig, request: RequestSpec, prepared: PreparedRequest
    ) -> InferenceResult:
        raise AdapterUnavailable(
            f"adapter {self.name!r} is an honest placeholder; implement and contract-test "
            f"the exact {route.provider}/{route.api_family} translation before live use"
        )

    async def close(self) -> None:
        return None


class AdapterFactory(Protocol):
    def __call__(
        self,
        *,
        http2: bool,
        connection_reuse: bool,
        transport_max_connections: int,
    ) -> Adapter: ...


@dataclass(frozen=True, slots=True)
class AdapterPlugin:
    """Credential-free adapter contract and immutable campaign identity declaration."""

    name: str
    version: str
    api_families: tuple[str, ...]
    transport_kind: str
    factory: AdapterFactory = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        valid_name = (
            isinstance(self.name, str)
            and bool(self.name)
            and self.name.replace("_", "").replace("-", "").isalnum()
        )
        if not valid_name:
            raise ValueError("adapter names must contain only letters, digits, '_' or '-'")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("adapter plugin version must be a nonempty string")
        if self.transport_kind not in {"http_json", "native", "custom"}:
            raise ValueError("adapter transport_kind must be http_json, native, or custom")
        if (
            not isinstance(self.api_families, tuple)
            or not self.api_families
            or any(not isinstance(item, str) or not item.strip() for item in self.api_families)
        ):
            raise ValueError("adapter api_families must be a nonempty tuple of names")
        normalized = tuple(sorted(set(self.api_families)))
        object.__setattr__(self, "api_families", normalized)
        if not callable(self.factory):
            raise TypeError("adapter factory must be callable")

    def public_identity(self) -> dict[str, Any]:
        declaration = {
            "schema_version": "adapter-plugin/v1",
            "name": self.name,
            "version": self.version,
            "api_families": list(self.api_families),
            "transport_kind": self.transport_kind,
        }
        return {**declaration, "identity_hash": sha256_json(declaration)}


ADAPTER_ENTRY_POINT_GROUP = "inference_endpoint_benchmark.adapters"
_ADAPTER_PLUGINS: dict[str, AdapterPlugin] = {}
_BUILTINS_REGISTERED = False
_ENTRY_POINTS_LOADED = False


def register_adapter(plugin: AdapterPlugin, *, replace: bool = False) -> None:
    """Register an adapter without editing the benchmark kernel.

    Third-party packages normally expose an :class:`AdapterPlugin` through the
    ``inference_endpoint_benchmark.adapters`` Python entry-point group. Direct registration is
    useful for private adapters and deterministic simulators.
    """

    if not isinstance(plugin, AdapterPlugin):
        raise TypeError("adapter registration requires an AdapterPlugin descriptor")
    if plugin.name in _ADAPTER_PLUGINS and not replace:
        raise ValueError(f"adapter already registered: {plugin.name}")
    _ADAPTER_PLUGINS[plugin.name] = plugin


def _register_builtins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .bedrock_converse import BedrockConverseAdapter
    from .openai_compatible import OpenAICompatibleAdapter
    from .providers import (
        AlibabaModelStudioAdapter,
        AlibabaModelStudioResponsesAdapter,
        AzureModelInferenceAdapter,
        AzureResponsesAdapter,
        BedrockMantleAdapter,
        BedrockMantleResponsesAdapter,
        OpenRouterAdapter,
        VertexOpenAIAdapter,
    )
    from .vertex_native import VertexNativeAdapter

    version = "0.1.0"
    builtins: tuple[AdapterPlugin, ...] = (
        AdapterPlugin(
            "openai_compatible",
            version,
            ("chat_completions",),
            "http_json",
            OpenAICompatibleAdapter,
        ),
        AdapterPlugin(
            "alibaba_model_studio",
            version,
            ("chat_completions",),
            "http_json",
            AlibabaModelStudioAdapter,
        ),
        AdapterPlugin(
            "alibaba_model_studio_responses",
            version,
            ("responses",),
            "http_json",
            AlibabaModelStudioResponsesAdapter,
        ),
        AdapterPlugin(
            "bedrock_mantle", version, ("chat_completions",), "http_json", BedrockMantleAdapter
        ),
        AdapterPlugin(
            "bedrock_mantle_responses",
            version,
            ("responses",),
            "http_json",
            BedrockMantleResponsesAdapter,
        ),
        AdapterPlugin(
            "azure_openai", version, ("chat_completions",), "http_json", AzureModelInferenceAdapter
        ),
        AdapterPlugin(
            "azure_model_inference",
            version,
            ("chat_completions",),
            "http_json",
            AzureModelInferenceAdapter,
        ),
        AdapterPlugin(
            "azure_responses", version, ("responses",), "http_json", AzureResponsesAdapter
        ),
        AdapterPlugin(
            "openrouter", version, ("chat_completions",), "http_json", OpenRouterAdapter
        ),
        AdapterPlugin(
            "vertex_openai", version, ("chat_completions",), "http_json", VertexOpenAIAdapter
        ),
        AdapterPlugin(
            "bedrock_converse",
            version,
            ("converse",),
            "native",
            BedrockConverseAdapter,
        ),
        AdapterPlugin(
            "bedrock_native",
            version,
            ("converse",),
            "native",
            BedrockConverseAdapter,
        ),
        AdapterPlugin(
            "vertex_native",
            version,
            ("chat_completions", "generate_content"),
            "native",
            VertexNativeAdapter,
        ),
        AdapterPlugin(
            "azure_model_inference_native",
            version,
            ("chat_completions",),
            "native",
            lambda **_: FailClosedAdapter("azure_model_inference_native"),
        ),
    )
    for plugin in builtins:
        register_adapter(plugin)
    _BUILTINS_REGISTERED = True


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    discovered = metadata.entry_points()
    selected = (
        discovered.select(group=ADAPTER_ENTRY_POINT_GROUP)
        if hasattr(discovered, "select")
        else discovered.get(ADAPTER_ENTRY_POINT_GROUP, ())
    )
    for entry_point in sorted(selected, key=lambda item: item.name):
        plugin = entry_point.load()
        if not isinstance(plugin, AdapterPlugin):
            raise AdapterUnavailable(
                f"adapter entry point {entry_point.name!r} must expose an AdapterPlugin"
            )
        if plugin.name != entry_point.name:
            raise AdapterUnavailable(
                f"adapter entry point {entry_point.name!r} declares mismatched name "
                f"{plugin.name!r}"
            )
        register_adapter(plugin)
    _ENTRY_POINTS_LOADED = True


def available_adapters() -> tuple[str, ...]:
    _register_builtins()
    _load_entry_points()
    return tuple(sorted(_ADAPTER_PLUGINS))


def adapter_plugin(name: str) -> AdapterPlugin:
    _register_builtins()
    _load_entry_points()
    plugin = _ADAPTER_PLUGINS.get(name)
    if plugin is None:
        available = ", ".join(sorted(_ADAPTER_PLUGINS)) or "none"
        raise AdapterUnavailable(f"unknown adapter {name!r}; available adapters: {available}")
    return plugin


def validate_adapter_route(route: RouteConfig) -> AdapterPlugin:
    """Resolve a route's adapter contract without credentials, networking, or output state."""

    plugin = adapter_plugin(route.adapter)
    if route.api_family not in plugin.api_families:
        supported = ", ".join(plugin.api_families)
        raise AdapterUnavailable(
            f"adapter {plugin.name!r} version {plugin.version!r} does not support "
            f"api_family={route.api_family!r}; supported: {supported}"
        )
    return plugin


def validate_adapter_instance(name: str, adapter: object) -> Adapter:
    required_methods = ("preflight", "prepare", "send_prepared", "close")
    missing = [
        method for method in required_methods if not callable(getattr(adapter, method, None))
    ]
    if missing:
        raise AdapterUnavailable(
            f"adapter factory {name!r} returned an invalid adapter; missing required methods: "
            + ", ".join(missing)
        )
    return adapter  # type: ignore[return-value]


def adapter_for(
    name: str,
    *,
    http2: bool = False,
    connection_reuse: bool = True,
    transport_max_connections: int = 256,
) -> Adapter:
    plugin = adapter_plugin(name)
    options: dict[str, Any] = {
        "http2": http2,
        "connection_reuse": connection_reuse,
        "transport_max_connections": transport_max_connections,
    }
    adapter = plugin.factory(**options)
    return validate_adapter_instance(name, adapter)
