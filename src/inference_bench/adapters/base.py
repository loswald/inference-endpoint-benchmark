from __future__ import annotations

from typing import Protocol

from ..models import InferenceResult, RequestSpec, RouteConfig


class AdapterUnavailable(RuntimeError):
    pass


class Adapter(Protocol):
    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult: ...

    async def close(self) -> None: ...


class FailClosedAdapter:
    def __init__(self, name: str) -> None:
        self.name = name

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
        raise AdapterUnavailable(
            f"adapter {self.name!r} is an honest placeholder; implement and contract-test "
            f"the exact {route.provider}/{route.api_family} translation before live use"
        )

    async def close(self) -> None:
        return None


def adapter_for(name: str) -> Adapter:
    from .openai_compatible import OpenAICompatibleAdapter

    if name in {"openai_compatible", "digitalocean", "azure_openai", "vertex_openai"}:
        return OpenAICompatibleAdapter()
    if name in {
        "bedrock_native",
        "vertex_native",
        "azure_model_inference_native",
        "openrouter",
    }:
        return FailClosedAdapter(name)
    raise AdapterUnavailable(f"unknown adapter: {name}")
