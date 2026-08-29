from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import replace

import pytest

from inference_bench.adapters import (
    AdapterPlugin,
    AdapterUnavailable,
    PreparedRequest,
    adapter_for,
    available_adapters,
    register_adapter,
)
from inference_bench.cli import run_campaign
from inference_bench.engine import BenchmarkEngine, deterministic_request_id
from inference_bench.ledger import Ledger
from inference_bench.models import InferenceResult, RequestSpec
from inference_bench.payload import (
    MaterializedPayload,
    materialize_openai_compatible,
    payload_binding_sha256,
)
from inference_bench.plan import build_plan


class ExampleProtocolAdapter:
    def __init__(self, **transport_options: object) -> None:
        self.transport_options = transport_options

    def preflight(self, route) -> None:  # type: ignore[no-untyped-def]
        return None

    def prepare(self, route, request):  # type: ignore[no-untyped-def]
        return PreparedRequest(payload=materialize_openai_compatible(route, request))

    async def infer(self, route, request):  # type: ignore[no-untyped-def]
        return InferenceResult(
            logical_id=request.logical_id,
            status="success",
            http_status=200,
            started_at_utc="2026-01-01T00:00:00Z",
            ended_at_utc="2026-01-01T00:00:00.010000Z",
            total_seconds=0.01,
            input_tokens=1,
            output_tokens=1,
        )

    async def send_prepared(self, route, request, prepared):  # type: ignore[no-untyped-def]
        return await self.infer(route, request)

    async def close(self) -> None:
        return None


def test_private_adapter_registers_without_editing_core_registry() -> None:
    name = "example_protocol_test"
    register_adapter(
        AdapterPlugin(
            name=name,
            version="1.2.3",
            api_families=("example_protocol",),
            transport_kind="custom",
            factory=ExampleProtocolAdapter,
        ),
        replace=True,
    )
    adapter = adapter_for(
        name,
        http2=True,
        connection_reuse=False,
        transport_max_connections=7,
    )
    assert isinstance(adapter, ExampleProtocolAdapter)
    assert adapter.transport_options == {
        "http2": True,
        "connection_reuse": False,
        "transport_max_connections": 7,
    }
    assert name in available_adapters()


def test_builtin_registry_contains_every_supported_provider_transport() -> None:
    assert {
        "openai_compatible",
        "bedrock_mantle",
        "bedrock_mantle_responses",
        "azure_model_inference",
        "azure_responses",
        "vertex_openai",
        "openrouter",
    } <= set(available_adapters())


def test_unknown_adapter_fails_before_run_output_is_created(tmp_path, campaign, route) -> None:
    unavailable = replace(route, adapter="uninstalled_private_protocol")
    config = replace(campaign, routes=(unavailable,))
    output = tmp_path / "must-not-exist"

    with pytest.raises(AdapterUnavailable, match="unknown adapter"):
        asyncio.run(run_campaign(config, output))

    assert not output.exists()


def test_incompatible_adapter_api_family_fails_during_plan(campaign, route) -> None:
    name = "example_incompatible_protocol_test"
    register_adapter(
        AdapterPlugin(
            name=name,
            version="1",
            api_families=("example_protocol",),
            transport_kind="custom",
            factory=ExampleProtocolAdapter,
        ),
        replace=True,
    )
    incompatible = replace(route, adapter=name, api_family="chat_completions")

    with pytest.raises(AdapterUnavailable, match="does not support api_family"):
        build_plan(replace(campaign, routes=(incompatible,)))


def test_infer_only_legacy_adapter_is_rejected_before_use() -> None:
    name = "example_infer_only_adapter_test"

    class InferOnlyAdapter:
        def __init__(self, **transport_options: object) -> None:
            self.transport_options = transport_options

        def preflight(self, route) -> None:  # type: ignore[no-untyped-def]
            return None

        async def infer(self, route, request):  # type: ignore[no-untyped-def]
            raise AssertionError("legacy infer() must never be called")

        async def close(self) -> None:
            return None

    register_adapter(
        AdapterPlugin(
            name=name,
            version="1",
            api_families=("example_protocol",),
            transport_kind="custom",
            factory=InferOnlyAdapter,
        ),
        replace=True,
    )

    with pytest.raises(AdapterUnavailable, match="prepare, send_prepared"):
        adapter_for(name)


def test_plugin_version_is_bound_into_campaign_identity(campaign, route) -> None:
    name = "example_versioned_protocol_test"

    def register(version: str) -> None:
        register_adapter(
            AdapterPlugin(
                name=name,
                version=version,
                api_families=("example_protocol",),
                transport_kind="custom",
                factory=ExampleProtocolAdapter,
            ),
            replace=True,
        )

    configured = replace(
        campaign,
        routes=(replace(route, adapter=name, api_family="example_protocol"),),
    )
    register("1")
    first_identity = configured.identity_hash
    first_public = configured.public_dict()["adapter_plugins"]
    register("2")

    assert configured.identity_hash != first_identity
    assert configured.public_dict()["adapter_plugins"] != first_public


def test_custom_protocol_exact_bytes_are_bound_to_ledger_and_cost(
    tmp_path, campaign, route
) -> None:
    name = "example_binary_protocol_test"
    exact_body = b"EXAMPLE/1\x00prompt=hello"
    generator_version = "example-binary/v1"
    wire_hash = hashlib.sha256(exact_body).hexdigest()
    bound_hash = payload_binding_sha256(exact_body, generator_version)
    token_upper_bound = 37
    created: list[BinaryProtocolAdapter] = []

    class BinaryProtocolAdapter:
        def __init__(self, **transport_options: object) -> None:
            self.transport_options = transport_options
            self.sent_bodies: list[bytes] = []

        def preflight(self, route) -> None:  # type: ignore[no-untyped-def]
            return None

        def prepare(self, route, request):  # type: ignore[no-untyped-def]
            return PreparedRequest(
                payload=MaterializedPayload(
                    value={"protocol": "EXAMPLE/1", "prompt": "hello"},
                    body=exact_body,
                    wire_body_sha256=wire_hash,
                    bound_payload_sha256=bound_hash,
                    input_token_upper_bound=token_upper_bound,
                    generator_version=generator_version,
                )
            )

        async def send_prepared(
            self, route, request, prepared  # type: ignore[no-untyped-def]
        ) -> InferenceResult:
            self.sent_bodies.append(prepared.payload.body)
            return InferenceResult(
                logical_id=request.logical_id,
                status="client_error",
                http_status=400,
                started_at_utc="2026-01-01T00:00:00Z",
                ended_at_utc="2026-01-01T00:00:00.010000Z",
                total_seconds=0.01,
            )

        async def close(self) -> None:
            return None

    def factory(**transport_options: object) -> BinaryProtocolAdapter:
        adapter = BinaryProtocolAdapter(**transport_options)
        created.append(adapter)
        return adapter

    register_adapter(
        AdapterPlugin(
            name=name,
            version="1",
            api_families=("example_protocol",),
            transport_kind="custom",
            factory=factory,
        ),
        replace=True,
    )
    custom_route = replace(route, adapter=name, api_family="example_protocol")
    config = replace(
        campaign,
        routes=(custom_route,),
        retries=0,
        input_token_reservation_factor=1.25,
    )
    spec = RequestSpec(
        logical_id="custom-protocol-request",
        route_id=custom_route.id,
        suite="protocol",
        cell_id="exact-bytes",
        messages=({"role": "user", "content": "ignored by proprietary materializer"},),
        planned_input_tokens=5,
        max_output_tokens=8,
        stream=False,
    )
    ledger = Ledger(tmp_path / "custom-protocol-ledger")
    ledger.initialize(campaign_hash=config.identity_hash, config_json="{}")

    async def run() -> tuple[
        InferenceResult | None, list[dict[str, object]], dict[str, object]
    ]:
        engine = BenchmarkEngine(config, ledger)
        result = await engine.execute(spec)
        rows = ledger.rows()
        claimed = next(event for event in ledger.event_rows() if event["kind"] == "request_claimed")
        await engine.close()
        return result, rows, claimed

    result, rows, claimed = asyncio.run(run())
    ledger.close()

    assert result is not None
    assert len(created) == 1
    assert created[0].sent_bodies == [exact_body]
    assert len(rows) == 1
    row = rows[0]
    expected_reserved_tokens = math.ceil(token_upper_bound * 1.25)
    expected_reservation = custom_route.worst_case_cost(expected_reserved_tokens, 8)
    assert row["request_id"] == deterministic_request_id(spec, 1, payload_hash=bound_hash)
    assert row["payload_sha256"] == bound_hash
    assert row["wire_body_sha256"] == wire_hash
    assert row["payload_generator_version"] == generator_version
    assert row["reserved_input_tokens"] == expected_reserved_tokens
    claimed_payload = json.loads(str(claimed["payload_json"]))
    assert claimed_payload["reserved_usd"] == pytest.approx(expected_reservation)
    assert row["reserved_usd"] == 0
    assert row["settled_usd"] == pytest.approx(expected_reservation)
    assert row["cost_basis"] == "reserved_upper_bound"
    assert result.cost_basis == "reserved_upper_bound"
