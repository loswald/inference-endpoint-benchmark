from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from inference_bench.adapters.embeddings import (
    embedding_adapter_plugin,
    materialize_embedding_request_for,
)
from inference_bench.adapters.vertex_embeddings import VertexEmbedContentAdapter
from inference_bench.embedding_benchmark import (
    build_embedding_plan,
    embedding_config_from_mapping,
    plan_embedding_requests,
    run_embedding_campaign,
)
from inference_bench.embedding_models import EmbeddingRequestSpec


class FakeCredentials:
    def __init__(self) -> None:
        self.valid = False
        self.token: str | None = None
        self.refresh_calls = 0

    def refresh(self, request: object) -> None:
        assert request == "refresh-transport"
        self.refresh_calls += 1
        self.valid = True
        self.token = "oauth-fixture"


def _profile() -> dict[str, object]:
    return {
        "schema": "embedding-benchmark/v1",
        "campaign": {
            "name": "vertex-embedding-contract-test",
            "max_cost_usd": 1,
            "concurrency": 4,
            "retries": 0,
        },
        "route": {
            "id": "vertex-gemini-embedding-2",
            "provider": "google-vertex-ai",
            "adapter": "vertex_embed_content",
            "model": "gemini-embedding-2",
            "base_url": (
                "https://aiplatform.us.rep.googleapis.com/v1/projects/benchmark-project/"
                "locations/us/publishers/google/models/gemini-embedding-2:embedContent"
            ),
            "auth": {
                "env": "GOOGLE_APPLICATION_CREDENTIALS",
                "header": "Authorization",
                "prefix": "Bearer ",
            },
            "region": "us",
            "api_family": "embeddings",
            "billing_channel": "vertex-paygo",
            "api_version": "v1",
            "model_version": "gemini-embedding-2",
            "quota_scope": "global-dynamic-shared-quota",
            "capabilities": {
                "max_input_tokens_per_item": 16,
                "max_batch_inputs": 1,
                "max_total_tokens_per_request": 16,
                "default_dimensions": 3,
                "supported_dimensions": [2, 3],
                "empty_input": "documented_invalid",
                "unicode_input": "documented_supported",
                "over_limit_input": "documented_truncate",
                "repeatability_cosine_minimum": 0.999999,
                "long_input_fraction": 0.75,
            },
            "input_usd_per_million": 0.1,
            "request_timeout_seconds": 10,
            "input_token_reservation_overhead": 2,
            "documentation_source_url": (
                "https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings"
            ),
            "pricing_source_url": "https://cloud.google.com/vertex-ai/pricing",
            "evidence_retrieved_at_utc": "2030-01-01T00:00:00Z",
            "evidence_bundle_sha256": "b" * 64,
        },
    }


def _adapter(
    client: httpx.AsyncClient,
    credentials: FakeCredentials | None = None,
) -> tuple[VertexEmbedContentAdapter, FakeCredentials]:
    credentials = credentials or FakeCredentials()
    return (
        VertexEmbedContentAdapter(
            client=client,
            credentials=credentials,
            auth_request_factory=lambda: "refresh-transport",
        ),
        credentials,
    )


def test_vertex_embed_content_is_typed_singleton_transport_and_materializes_exact_bytes() -> None:
    config = embedding_config_from_mapping(_profile())
    plugin = embedding_adapter_plugin(config.route.adapter)
    assert plugin.input_cardinality == "one"
    spec = EmbeddingRequestSpec(
        logical_id="vertex-embedding-request",
        route_id=config.route.id,
        cell_id="single",
        inputs=("never retain this",),
        planned_input_tokens=4,
        dimensions=2,
    )
    payload = materialize_embedding_request_for(config.route, spec)
    assert payload.value == {
        "content": {"role": "user", "parts": [{"text": "never retain this"}]},
        "embedContentConfig": {"outputDimensionality": 2},
    }
    assert payload.body == json.dumps(
        payload.value, sort_keys=True, separators=(",", ":")
    ).encode()
    assert payload.generator_version == "vertex-embed-content/v1"
    assert payload.wire_body_sha256 == hashlib.sha256(payload.body).hexdigest()


def test_vertex_singleton_plan_uses_distinct_repeat_calls_and_no_fake_batch_request() -> None:
    config = embedding_config_from_mapping(_profile())
    specs = plan_embedding_requests(config.route)
    cells = {spec.cell_id for spec in specs}
    assert {"repeatability-call-1", "repeatability-call-2"} <= cells
    assert "repeatability-pair" not in cells
    assert not any(cell.startswith("batch-") for cell in cells)
    assert "invalid-batch-over-documented-max" not in cells
    plan = build_embedding_plan(config).public_dict()
    assert plan["adapter_identity"]["input_cardinality"] == "one"
    assert plan["derived_cells"] == [
        {
            "cell_id": "repeatability-exact-across-requests",
            "kind": "cross_request_exact_vector_equality",
            "member_logical_ids": [
                next(spec.logical_id for spec in specs if spec.cell_id == "repeatability-call-1"),
                next(spec.logical_id for spec in specs if spec.cell_id == "repeatability-call-2"),
            ],
        }
    ]


def test_vertex_embed_content_parses_usage_vector_and_hashes_request_identity() -> None:
    config = embedding_config_from_mapping(_profile())
    observed: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.content)
        assert request.headers["authorization"] == "Bearer oauth-fixture"
        return httpx.Response(
            200,
            headers={"x-goog-request-id": "private-google-request-id"},
            json={
                "embedding": {"values": [1.0, 0.0, 0.0]},
                "usageMetadata": {"promptTokenCount": 4, "totalTokenCount": 4},
                "truncated": False,
            },
        )

    async def run() -> tuple[object, object, FakeCredentials]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, credentials = _adapter(client)
        spec = EmbeddingRequestSpec(
            logical_id="vertex-embedding-request",
            route_id=config.route.id,
            cell_id="single",
            inputs=("private text",),
            planned_input_tokens=4,
            dimensions=3,
        )
        prepared = adapter.prepare(config.route, spec)
        result = await adapter.send_prepared(config.route, spec, prepared)
        await client.aclose()
        return prepared, result, credentials

    prepared, result, credentials = asyncio.run(run())
    assert observed == [prepared.payload.body]
    assert credentials.refresh_calls == 1
    assert result.validation_passed
    assert result.prompt_tokens == 4
    assert result.total_tokens == 4
    assert result.vectors[0].dimensions == 3
    assert result.provider_request_id_sha256 == hashlib.sha256(
        b"private-google-request-id"
    ).hexdigest()
    public = json.dumps(result.public_dict(), sort_keys=True)
    assert "private text" not in public
    assert "private-google-request-id" not in public
    assert "oauth-fixture" not in public
    assert "[1.0, 0.0, 0.0]" not in public


def test_vertex_embedding_campaign_is_complete_resumable_and_privacy_safe(
    tmp_path: Path,
) -> None:
    config = embedding_config_from_mapping(_profile())
    send_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal send_count
        send_count += 1
        payload = json.loads(request.content)
        text = payload["content"]["parts"][0]["text"]
        if not text:
            return httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT"}})
        dimensions = payload["embedContentConfig"]["outputDimensionality"]
        truncated = text.count(" token") >= 16
        return httpx.Response(
            200,
            json={
                "embedding": {"values": [1.0] + [0.0] * (dimensions - 1)},
                "usageMetadata": {"promptTokenCount": 1, "totalTokenCount": 1},
                "truncated": truncated,
            },
        )

    async def run_twice() -> tuple[dict[str, object], dict[str, object], int]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, _ = _adapter(client)
        first = await run_embedding_campaign(config, tmp_path, adapter=adapter)
        first_count = send_count
        second_adapter, _ = _adapter(client, FakeCredentials())
        second = await run_embedding_campaign(config, tmp_path, adapter=second_adapter)
        await client.aclose()
        return first, second, first_count

    first, second, first_count = asyncio.run(run_twice())
    assert send_count == first_count
    assert first["coverage"] == second["coverage"]
    assert first["state_counts"] == {"passed": first["planned_cells"]}
    derived = next(
        row
        for row in first["coverage"]
        if row["cell_id"] == "repeatability-exact-across-requests"
    )
    assert derived["result"]["vector_hashes_equal"] is True
    public = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            tmp_path / "embedding-plan.json",
            tmp_path / "embedding-events.jsonl",
            tmp_path / "embedding-report.json",
            tmp_path / "embedding-report.md",
        )
    )
    assert "Portable embedding benchmark" not in public
    assert "oauth-fixture" not in public
    assert '"values"' not in public


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_url", "https://aiplatform.googleapis.com/v1/wrong", "embedContent"),
        ("provider", "not-google", "google-vertex-ai"),
    ],
)
def test_vertex_embedding_route_contract_fails_before_credentials(
    field: str, replacement: str, message: str
) -> None:
    profile = _profile()
    route = profile["route"]
    assert isinstance(route, dict)
    route[field] = replacement
    with pytest.raises(ValueError, match=message):
        embedding_config_from_mapping(profile)
