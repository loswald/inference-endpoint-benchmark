from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from inference_bench.cli import main
from inference_bench.config import load_config
from inference_bench.plan import build_plan
from inference_bench.profile_config import (
    compile_profile_files,
    compose_profile_config,
    load_profile_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _provider_profile() -> dict[str, object]:
    return {
        "schema": "provider-profile/v1",
        "provider": "fictional-cloud",
        "route_defaults": {
            "adapter": "openai_compatible",
            "base_url": "https://api.fictional.example.test/v1/chat/completions",
            "auth": {
                "env": "FICTIONAL_API_KEY",
                "header": "Authorization",
                "prefix": "Bearer ",
            },
            "region": "fictional-region-1",
            "api_family": "chat_completions",
            "billing_channel": "startup_credits",
            "api_version": "v1",
            "model_version": "catalog-2030-01-01",
            "quota_scope": "fictional-account-scope",
            "context_tokens": 32_768,
            "max_output_tokens": 4_096,
            "stream_usage_mode": "required",
            "request_timeout_seconds": 60,
            "transport_max_connections": 8,
            "input_usd_per_million": 0.1,
            "output_usd_per_million": 0.2,
            "documentation_source_url": "https://docs.fictional.example.test/inference",
            "pricing_source_url": "https://docs.fictional.example.test/pricing",
            "evidence_retrieved_at_utc": "2030-01-01T00:00:00Z",
            "evidence_bundle_sha256": "a" * 64,
            "capabilities": {
                "documentation_checked_utc": "2030-01-01T00:00:00Z",
                "streaming": True,
            },
        },
        "routes": [
            {"id": "zeta-chat", "model": "zeta-1", "capabilities": {"vision": False}},
            {"id": "alpha-chat", "model": "alpha-2", "capabilities": {"vision": True}},
            {"id": "unused-chat", "model": "unused-3"},
        ],
    }


def _experiment_profile() -> dict[str, object]:
    return {
        "schema": "benchmark-experiment/v1",
        "campaign": {
            "name": "fictional-portability-test",
            "seed": 17,
            "max_wall_seconds": 600,
            "max_cost_usd": 20,
            "launch_reserve_seconds": 30,
            "launch_reserve_usd": 1,
            "concurrency": 4,
            "retries": 1,
            "client_location": "fictional-client-region",
        },
        "route_selection": {"include": ["zeta-chat", "alpha-chat"]},
        "route_overrides": {
            "zeta-chat": {
                "auth": {"env": "FICTIONAL_SECONDARY_API_KEY"},
                "request_timeout_seconds": 90,
                "capabilities": {"caching": "supported"},
            }
        },
        "suites": {"static": {"enabled": True, "offered_rps": 1}},
    }


def _write_yaml(path: Path, values: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def test_profile_composition_is_deterministic_and_uses_normal_config_validation(
    tmp_path: Path,
) -> None:
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"
    first = tmp_path / "compiled-a.yaml"
    second = tmp_path / "compiled-b.yaml"
    _write_yaml(provider, _provider_profile())
    _write_yaml(experiment, _experiment_profile())

    one = compile_profile_files(provider, experiment, first)
    two = compile_profile_files(provider, experiment, second)

    assert first.read_bytes() == second.read_bytes()
    assert one.compiled_sha256 == two.compiled_sha256
    assert one.config.identity_hash == load_config(first).identity_hash
    assert [route.id for route in one.config.routes] == ["alpha-chat", "zeta-chat"]
    assert all(route.provider == "fictional-cloud" for route in one.config.routes)

    zeta = next(route for route in one.config.routes if route.id == "zeta-chat")
    assert zeta.auth.env == "FICTIONAL_SECONDARY_API_KEY"
    assert zeta.auth.header == "Authorization"
    assert zeta.auth.prefix == "Bearer "
    assert zeta.request_timeout_seconds == 90
    assert zeta.capabilities == {
        "documentation_checked_utc": "2030-01-01T00:00:00Z",
        "streaming": True,
        "vision": False,
        "caching": "supported",
    }


def test_compile_profile_cli_emits_hashes_and_canonical_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"
    output = tmp_path / "compiled.yaml"
    _write_yaml(provider, _provider_profile())
    _write_yaml(experiment, _experiment_profile())

    assert main(
        [
            "compile-profile",
            str(provider),
            str(experiment),
            "--output",
            str(output),
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["config"] == str(output)
    assert len(result["compiled_sha256"]) == 64
    assert len(result["config_identity_sha256"]) == 64
    assert load_config(output).name == "fictional-portability-test"


def test_profile_schema_rejects_unknown_catalog_fields_even_when_route_is_unselected() -> None:
    provider = _provider_profile()
    routes = provider["routes"]
    assert isinstance(routes, list)
    assert isinstance(routes[-1], dict)
    routes[-1]["one_time_provider_hack"] = True

    with pytest.raises(ValueError, match="one_time_provider_hack"):
        compose_profile_config(provider, _experiment_profile())


def test_profile_schema_validates_unselected_catalog_routes() -> None:
    provider = _provider_profile()
    routes = provider["routes"]
    assert isinstance(routes, list)
    assert isinstance(routes[-1], dict)
    routes[-1]["auth"] = "embedded-secret"

    with pytest.raises(ValueError, match="route.auth must be a string-keyed mapping"):
        compose_profile_config(provider, _experiment_profile())


def test_profile_selection_and_overrides_fail_closed() -> None:
    experiment = _experiment_profile()
    selection = experiment["route_selection"]
    assert isinstance(selection, dict)
    selection["include"] = ["missing-route"]
    with pytest.raises(ValueError, match="unknown route"):
        compose_profile_config(_provider_profile(), experiment)

    experiment = _experiment_profile()
    overrides = experiment["route_overrides"]
    assert isinstance(overrides, dict)
    overrides["zeta-chat"] = {"provider": "different-provider"}
    with pytest.raises(ValueError, match="profile-controlled field"):
        compose_profile_config(_provider_profile(), experiment)


def test_profile_loader_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"
    provider.write_text(
        "schema: provider-profile/v1\nprovider: first\nprovider: second\nroutes: []\n",
        encoding="utf-8",
    )
    _write_yaml(experiment, _experiment_profile())

    with pytest.raises(ValueError, match="duplicate mapping key"):
        load_profile_config(provider, experiment)


def test_compiled_output_cannot_overwrite_an_input(tmp_path: Path) -> None:
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"
    _write_yaml(provider, _provider_profile())
    _write_yaml(experiment, _experiment_profile())

    with pytest.raises(ValueError, match="cannot overwrite"):
        compile_profile_files(provider, experiment, provider)


def test_public_alibaba_profile_compiles_without_resolving_credentials(tmp_path: Path) -> None:
    compilation = compile_profile_files(
        ROOT / "examples" / "provider-profiles" / "alibaba-model-studio-singapore.yaml",
        ROOT / "examples" / "experiment-profiles" / "standard-static-and-capacity.yaml",
        tmp_path / "alibaba.yaml",
    )

    routes = {route.id: route for route in compilation.config.routes}
    assert set(routes) == {
        "alibaba-sg-qwen3.8-flash",
        "alibaba-sg-qwen3.8-max",
        "alibaba-sg-qwen3.8-27b",
        "alibaba-sg-deepseek-v4-flash",
        "alibaba-sg-deepseek-v4-pro",
        "alibaba-sg-kimi-k2.7-code",
        "alibaba-sg-kimi-k3",
        "alibaba-sg-glm-5.2",
    }
    assert all(route.adapter == "alibaba_model_studio" for route in routes.values())
    assert all(route.billing_channel == "pay_as_you_go" for route in routes.values())
    assert all(route.auth.env == "DASHSCOPE_API_KEY" for route in routes.values())


def test_public_azure_foundry_profile_is_exactly_the_five_live_proved_text_routes(
    tmp_path: Path,
) -> None:
    compilation = compile_profile_files(
        ROOT / "examples" / "provider-profiles" / "azure-ai-foundry-eastus2.yaml",
        ROOT / "examples" / "experiment-profiles" / "standard-static-and-capacity.yaml",
        tmp_path / "azure.yaml",
    )

    routes = {route.id: route for route in compilation.config.routes}
    assert set(routes) == {
        "azure-gpt-5.6-sol-eastus2-responses",
        "azure-gpt-5.6-terra-eastus2-responses",
        "azure-gpt-5.6-luna-eastus2-responses",
        "azure-deepseek-v4-flash-eastus2-chat",
        "azure-kimi-k2.6-eastus2-chat",
    }
    assert all(route.region == "eastus2" for route in routes.values())
    assert all(route.auth.env == "AZURE_INFERENCE_CREDENTIAL" for route in routes.values())
    assert all(route.auth.header == "api-key" for route in routes.values())
    assert all("replace-with-resource-name" in route.base_url for route in routes.values())

    gpt_routes = [route for route in routes.values() if route.model.startswith("gpt-5.6-")]
    assert len(gpt_routes) == 3
    assert all(route.adapter == "azure_responses" for route in gpt_routes)
    assert all(route.api_family == "responses" for route in gpt_routes)
    assert all(route.model_version == "2026-07-09" for route in gpt_routes)
    assert all(route.context_tokens == 272_000 for route in gpt_routes)
    assert all(route.max_output_tokens == 128_000 for route in gpt_routes)

    assert routes["azure-deepseek-v4-flash-eastus2-chat"].adapter == "azure_model_inference"
    assert routes["azure-deepseek-v4-flash-eastus2-chat"].model_version == "2026-04-23"
    assert routes["azure-kimi-k2.6-eastus2-chat"].adapter == "azure_model_inference"
    assert routes["azure-kimi-k2.6-eastus2-chat"].model_version == "2026-04-20"
    assert all("claude" not in route.model.casefold() for route in routes.values())
    assert all("router" not in route.model.casefold() for route in routes.values())
    assert all("embedding" not in route.model.casefold() for route in routes.values())

    evidence = ROOT / "docs" / "provider-contracts" / "azure-ai-foundry-eastus2-2026-08-29.yaml"
    evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert all(route.evidence_bundle_sha256 == evidence_sha for route in routes.values())


def test_public_bedrock_profile_is_the_one_exact_live_proved_mantle_route(
    tmp_path: Path,
) -> None:
    provider_path = (
        ROOT / "examples" / "provider-profiles" / "amazon-bedrock-mantle-us-east-1.yaml"
    )
    compilation = compile_profile_files(
        provider_path,
        ROOT / "examples" / "experiment-profiles" / "standard-static-and-capacity.yaml",
        tmp_path / "bedrock.yaml",
    )

    assert [route.id for route in compilation.config.routes] == [
        "aws-bedrock-us-east-1-zai-glm-4.7-flash"
    ]
    route = compilation.config.routes[0]
    assert route.adapter == "bedrock_mantle"
    assert route.model == "zai.glm-4.7-flash"
    assert route.region == "us-east-1"
    assert route.api_family == "chat_completions"
    assert route.base_url == "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
    assert route.auth.env == "BEDROCK_API_KEY"
    assert route.context_tokens == 203_000
    assert route.max_output_tokens == 4_000
    assert route.input_usd_per_million == 0.07
    assert route.output_usd_per_million == 0.40

    contract_path = (
        ROOT / "docs" / "provider-contracts" / "amazon-bedrock-mantle-glm47-flash-2026-08-29.yaml"
    )
    assert route.evidence_bundle_sha256 == hashlib.sha256(contract_path.read_bytes()).hexdigest()

    provider = yaml.safe_load(provider_path.read_text(encoding="utf-8"))
    experiment = {
        "schema": "benchmark-experiment/v1",
        "campaign": {
            "name": "bedrock-credential-free-plan-test",
            "seed": 17,
            "max_wall_seconds": 600,
            "max_cost_usd": 20,
            "launch_reserve_seconds": 30,
            "launch_reserve_usd": 1,
            "concurrency": 4,
            "retries": 0,
            "client_location": "test-client-region",
        },
        "route_selection": {"include": [route.id]},
        "suites": {
            "warmup": {"enabled": True, "repeats": 1, "shapes": ["short_short"]}
        },
    }
    plan = build_plan(compose_profile_config(provider, experiment).config)
    assert plan.static_requests == 1
    assert plan.native_placeholder_routes == ()


def test_public_vertex_profile_is_the_exact_live_proved_global_openai_route(
    tmp_path: Path,
) -> None:
    contract_path = (
        ROOT / "docs" / "provider-contracts" / "google-vertex-gemini36-flash-global-2026-08-29.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    assert contract["runnable_in_library"] is True
    assert contract["route_contract"]["api_family"] == "openai_compatible_chat_completions_v1"
    assert contract["live_proof"]["evidence_scope"] == "transport_only_pre_final_build"

    provider_path = (
        ROOT / "examples" / "provider-profiles" / "google-vertex-gemini36-flash-global.yaml"
    )
    compilation = compile_profile_files(
        provider_path,
        ROOT / "examples" / "experiment-profiles" / "standard-static-and-capacity.yaml",
        tmp_path / "vertex.yaml",
    )
    assert [route.id for route in compilation.config.routes] == [
        "google-vertex-global-gemini-3.6-flash"
    ]
    route = compilation.config.routes[0]
    assert route.adapter == "vertex_openai"
    assert route.model == "google/gemini-3.6-flash"
    assert route.region == "global"
    assert route.context_tokens == 1_048_576
    assert route.max_output_tokens == 65_536
    assert route.input_usd_per_million == 0.75
    assert route.cached_input_usd_per_million == 0.075
    assert route.output_usd_per_million == 3.75
    assert route.evidence_bundle_sha256 == hashlib.sha256(contract_path.read_bytes()).hexdigest()

    provider = yaml.safe_load(provider_path.read_text(encoding="utf-8"))
    experiment = {
        "schema": "benchmark-experiment/v1",
        "campaign": {
            "name": "vertex-credential-free-plan-test",
            "seed": 17,
            "max_wall_seconds": 600,
            "max_cost_usd": 20,
            "launch_reserve_seconds": 30,
            "launch_reserve_usd": 1,
            "concurrency": 4,
            "retries": 0,
            "client_location": "test-client-region",
        },
        "route_selection": {"include": [route.id]},
        "suites": {
            "warmup": {"enabled": True, "repeats": 1, "shapes": ["short_short"]}
        },
    }
    plan = build_plan(compose_profile_config(provider, experiment).config)
    assert plan.static_requests == 1
    assert plan.native_placeholder_routes == ()
