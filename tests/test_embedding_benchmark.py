from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from inference_bench.adapters.embeddings import (
    OpenAICompatibleEmbeddingsAdapter,
)
from inference_bench.embedding_benchmark import (
    build_embedding_plan,
    embedding_config_from_mapping,
    load_embedding_config,
    plan_embedding_canary,
    plan_embedding_requests,
    run_embedding_campaign,
)
from inference_bench.embedding_models import EmbeddingRequestSpec

ROOT = Path(__file__).resolve().parents[1]


def _profile() -> dict[str, object]:
    return {
        "schema": "embedding-benchmark/v1",
        "campaign": {
            "name": "embedding-contract-test",
            "max_cost_usd": 1,
            "concurrency": 4,
            "retries": 0,
        },
        "route": {
            "id": "fictional-embedding",
            "provider": "fictional-provider",
            "adapter": "openai_compatible_embeddings",
            "model": "embedding-model",
            "base_url": "https://embedding.example.test/v1/embeddings",
            "auth": {
                "env": "FICTIONAL_EMBEDDING_API_KEY",
                "header": "Authorization",
                "prefix": "Bearer ",
            },
            "region": "test-region",
            "api_family": "embeddings",
            "billing_channel": "test",
            "api_version": "v1",
            "model_version": "immutable-test-model",
            "quota_scope": "test-account",
            "capabilities": {
                "max_input_tokens_per_item": 16,
                "max_batch_inputs": 4,
                "max_total_tokens_per_request": 64,
                "default_dimensions": 3,
                "supported_dimensions": [2, 3],
                "empty_input": "documented_invalid",
                "unicode_input": "documented_supported",
                "repeatability_cosine_minimum": 0.999999,
                "long_input_fraction": 0.75,
            },
            "input_usd_per_million": 0.1,
            "request_timeout_seconds": 10,
            "input_token_reservation_overhead": 2,
            "documentation_source_url": "https://docs.example.test/embeddings",
            "pricing_source_url": "https://docs.example.test/pricing",
            "evidence_retrieved_at_utc": "2030-01-01T00:00:00Z",
            "evidence_bundle_sha256": "a" * 64,
        },
    }


def test_public_embedding_profiles_bind_exact_models_and_documented_limits() -> None:
    azure = load_embedding_config(
        ROOT / "examples/embedding-profiles/azure-text-embedding-3-large.yaml"
    )
    alibaba = load_embedding_config(
        ROOT / "examples/embedding-profiles/alibaba-text-embedding-v4-singapore.yaml"
    )
    vertex = load_embedding_config(
        ROOT / "examples/embedding-profiles/vertex-gemini-embedding-2-us.template.yaml"
    )

    assert azure.route.model == "text-embedding-3-large"
    assert azure.route.api_family == "embeddings"
    assert azure.route.capabilities.max_input_tokens_per_item == 8192
    assert azure.route.capabilities.max_batch_inputs == 2048
    assert azure.route.capabilities.default_dimensions == 3072
    assert alibaba.route.model == "text-embedding-v4"
    assert alibaba.route.capabilities.max_input_tokens_per_item == 8192
    assert alibaba.route.capabilities.max_batch_inputs == 10
    assert alibaba.route.capabilities.supported_dimensions == (
        64,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
    )
    assert vertex.route.model == "gemini-embedding-2"
    assert vertex.route.adapter == "vertex_embed_content"
    assert vertex.route.region == "us"
    assert vertex.route.capabilities.max_input_tokens_per_item == 8192
    assert vertex.route.capabilities.default_dimensions == 3072
    assert vertex.route.input_usd_per_million == 0.2
    evidence = ROOT / "evidence/vertex-gemini-embedding-2-2026-08-30.json"
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == (
        vertex.route.evidence_bundle_sha256
    )


def test_embedding_api_family_and_adapter_fail_closed_at_plan_boundary() -> None:
    wrong_family = _profile()
    route = wrong_family["route"]
    assert isinstance(route, dict)
    route["api_family"] = "chat_completions"
    with pytest.raises(ValueError, match="api_family=embeddings"):
        embedding_config_from_mapping(wrong_family)

    wrong_adapter = _profile()
    route = wrong_adapter["route"]
    assert isinstance(route, dict)
    route["adapter"] = "chat_adapter"
    with pytest.raises(ValueError, match="unknown embedding adapter"):
        embedding_config_from_mapping(wrong_adapter)


def test_embedding_plan_covers_limits_batches_unicode_invalids_and_privacy() -> None:
    config = embedding_config_from_mapping(_profile())
    specs = plan_embedding_requests(config.route)
    cells = {spec.cell_id for spec in specs}
    assert {
        "single-short",
        "unicode-multilingual",
        "repeatability-pair",
        "single-near-documented-limit",
        "batch-2",
        "batch-4",
        "dimensions-2",
        "invalid-empty-input",
        "invalid-batch-over-documented-max",
        "item-over-documented-max",
    } <= cells
    near = next(spec for spec in specs if spec.cell_id == "single-near-documented-limit")
    assert near.planned_input_tokens == 12
    assert len(next(spec for spec in specs if spec.cell_id == "batch-4").inputs) == 4

    plan = build_embedding_plan(config).public_dict()
    serialized = json.dumps(plan, ensure_ascii=False)
    assert "Portable embedding benchmark" not in serialized
    assert "多语言检索" not in serialized
    assert plan["privacy"] == {
        "input_text_retained": False,
        "embedding_vectors_retained": False,
        "input_and_vector_sha256_only": True,
    }
    assert all(len(row["wire_body_sha256"]) == 64 for row in plan["requests"])
    assert all(len(row["bound_payload_sha256"]) == 64 for row in plan["requests"])


def test_embedding_canary_is_one_distinct_privacy_safe_admission_request() -> None:
    config = embedding_config_from_mapping(_profile())
    specs = plan_embedding_canary(config.route)
    assert len(specs) == 1
    assert specs[0].cell_id == "admission-canary"
    assert specs[0].logical_id not in {
        spec.logical_id for spec in plan_embedding_requests(config.route)
    }
    plan = build_embedding_plan(config, plan_kind="canary").public_dict()
    assert plan["plan_kind"] == "canary"
    assert plan["request_count"] == 1
    assert plan["requests"][0]["cell_id"] == "admission-canary"
    assert "Portable embedding admission canary" not in json.dumps(plan)


def test_adapter_sends_exact_prepared_bytes_and_retains_no_vectors_or_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = embedding_config_from_mapping(_profile())
    monkeypatch.setenv("FICTIONAL_EMBEDDING_API_KEY", "test-only-secret")
    observed_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_bodies.append(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "private-provider-id"},
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0, 2.0, 3.0]},
                    {"object": "embedding", "index": 1, "embedding": [1.0, 2.0, 3.0]},
                ],
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleEmbeddingsAdapter(client=client)
    spec = EmbeddingRequestSpec(
        logical_id="repeatability",
        route_id=config.route.id,
        cell_id="repeatability-pair",
        inputs=("never retain this", "never retain this"),
        planned_input_tokens=8,
        dimensions=3,
        input_encoding="string_array",
        repeatability_pairs=((0, 1),),
    )
    prepared = adapter.prepare(config.route, spec)
    result = asyncio.run(adapter.send_prepared(config.route, spec, prepared))
    asyncio.run(client.aclose())

    assert observed_bodies == [prepared.payload.body]
    assert result.validation_passed
    assert [value.dimensions for value in result.vectors] == [3, 3]
    assert result.repeatability[0].cosine_similarity == pytest.approx(1.0)
    assert result.repeatability[0].passed
    public = json.dumps(result.public_dict(), sort_keys=True)
    assert "never retain this" not in public
    assert "private-provider-id" not in public
    assert '"embedding"' not in public
    assert all(len(value.vector_sha256) == 64 for value in result.vectors)


def test_adapter_surfaces_usage_dimension_and_norm_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = embedding_config_from_mapping(_profile())
    monkeypatch.setenv("FICTIONAL_EMBEDDING_API_KEY", "test-only-secret")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.0, 0.0]}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleEmbeddingsAdapter(client=client)
    spec = EmbeddingRequestSpec(
        logical_id="invalid-success",
        route_id=config.route.id,
        cell_id="single",
        inputs=("x",),
        planned_input_tokens=1,
        dimensions=3,
    )
    result = asyncio.run(
        adapter.send_prepared(config.route, spec, adapter.prepare(config.route, spec))
    )
    asyncio.run(client.aclose())

    assert result.status == "success"
    assert not result.validation_passed
    assert {
        "usage_missing_or_invalid",
        "embedding_vector_zero_or_invalid_norm",
        "embedding_dimension_mismatch",
    } <= set(result.validation_errors)


def test_plan_run_receipts_report_and_resume_are_complete_and_privacy_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = embedding_config_from_mapping(_profile())
    monkeypatch.setenv("FICTIONAL_EMBEDDING_API_KEY", "test-only-secret")
    send_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal send_count
        send_count += 1
        payload = json.loads(request.content)
        raw_input = payload["input"]
        inputs = raw_input if isinstance(raw_input, list) else [raw_input]
        if any(not value for value in inputs) or len(inputs) > 4:
            return httpx.Response(400, json={"error": {"type": "invalid_request"}})
        if any(value.count(" token") >= 16 for value in inputs):
            return httpx.Response(400, json={"error": {"type": "invalid_request"}})
        dimensions = payload.get("dimensions", 3)
        vector = [1.0] + [0.0] * (dimensions - 1)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": vector}
                    for index, _value in enumerate(inputs)
                ],
                "usage": {
                    "prompt_tokens": max(1, len(inputs)),
                    "total_tokens": max(1, len(inputs)),
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleEmbeddingsAdapter(client=client)
    first = asyncio.run(run_embedding_campaign(config, tmp_path, adapter=adapter))
    first_send_count = send_count
    second = asyncio.run(run_embedding_campaign(config, tmp_path, adapter=adapter))
    asyncio.run(client.aclose())

    assert first == second or first["coverage"] == second["coverage"]
    assert send_count == first_send_count
    assert first["planned_cells"] == len(plan_embedding_requests(config.route))
    assert first["state_counts"] == {"passed": first["planned_cells"]}
    assert first["settled_cost_usd"] > 0
    public_artifacts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            tmp_path / "embedding-plan.json",
            tmp_path / "embedding-events.jsonl",
            tmp_path / "embedding-report.json",
            tmp_path / "embedding-report.md",
        )
    )
    assert "Portable embedding benchmark" not in public_artifacts
    assert "多语言检索" not in public_artifacts
    assert "test-only-secret" not in public_artifacts
    assert '"embedding":[' not in public_artifacts
    assert "not_run" not in first["state_counts"]


def test_canary_run_sends_exactly_one_request_and_resumes_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = embedding_config_from_mapping(_profile())
    monkeypatch.setenv("FICTIONAL_EMBEDDING_API_KEY", "test-only-secret")
    sends = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}],
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleEmbeddingsAdapter(client=client)
    first = asyncio.run(
        run_embedding_campaign(config, tmp_path, adapter=adapter, plan_kind="canary")
    )
    second = asyncio.run(
        run_embedding_campaign(config, tmp_path, adapter=adapter, plan_kind="canary")
    )
    asyncio.run(client.aclose())

    assert sends == 1
    assert first["plan_kind"] == second["plan_kind"] == "canary"
    assert first["planned_cells"] == 1
    assert first["state_counts"] == {"passed": 1}
    assert "not benchmark coverage" in (tmp_path / "embedding-report.md").read_text(
        encoding="utf-8"
    )
