from __future__ import annotations

from typing import Protocol

from ..models import InferenceResult, RequestSpec, RouteConfig


class AdapterUnavailable(RuntimeError):
    pass


class Adapter(Protocol):
    def preflight(self, route: RouteConfig) -> None: ...

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult: ...

    async def close(self) -> None: ...


class FailClosedAdapter:
    def __init__(self, name: str) -> None:
        self.name = name

    def preflight(self, route: RouteConfig) -> None:
        raise AdapterUnavailable(
            f"adapter {self.name!r} is an honest placeholder for {route.provider}"
        )

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
        raise AdapterUnavailable(
            f"adapter {self.name!r} is an honest placeholder; implement and contract-test "
            f"the exact {route.provider}/{route.api_family} translation before live use"
        )

    async def close(self) -> None:
        return None


def adapter_for(
    name: str,
    *,
    http2: bool = False,
    connection_reuse: bool = True,
    transport_max_connections: int = 256,
) -> Adapter:
    from .openai_compatible import OpenAICompatibleAdapter
    from .providers import (
        AzureModelInferenceAdapter,
        AzureResponsesAdapter,
        BedrockMantleAdapter,
        BedrockMantleResponsesAdapter,
        OpenRouterAdapter,
        VertexOpenAIAdapter,
    )

    if name in {"openai_compatible", "digitalocean"}:
        return OpenAICompatibleAdapter(
            http2=http2,
            connection_reuse=connection_reuse,
            transport_max_connections=transport_max_connections,
        )
    provider_adapters = {
        "bedrock_mantle": BedrockMantleAdapter,
        "bedrock_mantle_responses": BedrockMantleResponsesAdapter,
        "azure_openai": AzureModelInferenceAdapter,
        "azure_model_inference": AzureModelInferenceAdapter,
        "azure_responses": AzureResponsesAdapter,
        "openrouter": OpenRouterAdapter,
        "vertex_openai": VertexOpenAIAdapter,
    }
    if name in provider_adapters:
        return provider_adapters[name](
            http2=http2,
            connection_reuse=connection_reuse,
            transport_max_connections=transport_max_connections,
        )
    if name in {
        "bedrock_native",
        "vertex_native",
        "azure_model_inference_native",
    }:
        return FailClosedAdapter(name)
    raise AdapterUnavailable(f"unknown adapter: {name}")
