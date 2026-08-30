from __future__ import annotations

import copy
from typing import Any

import pytest

from inference_bench.profile_config import compose_profile_config

AS_OF_UTC = "2026-08-30T00:00:00Z"


def _provider_profile_v2() -> dict[str, Any]:
    return {
        "schema": "provider-profile/v2",
        "provider": "portable-cloud",
        "catalog": {
            "documentation_checked_at_utc": "2026-08-24T00:00:00Z",
            "revalidated_at_utc": "2026-08-25T00:00:00Z",
            "freshness_window_days": 30,
        },
        "route_defaults": {
            "adapter": "openai_compatible",
            "base_url": "https://api.portable.example.test/v1/chat/completions",
            "auth": {
                "env": "PORTABLE_API_KEY",
                "header": "Authorization",
                "prefix": "Bearer ",
            },
            "region": "portable-region-1",
            "api_family": "chat_completions",
            "billing_channel": "startup_credits",
            "api_version": "v1",
            "model_version": "2026-08",
            "quota_scope": "portable-account",
            "context_tokens": 32_768,
            "max_output_tokens": 4_096,
            "stream_usage_mode": "required",
            "request_timeout_seconds": 60,
            "transport_max_connections": 8,
            "input_usd_per_million": 0.1,
            "output_usd_per_million": 0.2,
            "documentation_source_url": "https://docs.portable.example.test/models",
            "pricing_source_url": "https://docs.portable.example.test/pricing",
            "evidence_retrieved_at_utc": "2026-08-25T00:00:00Z",
            "evidence_bundle_sha256": "a" * 64,
            "capabilities": {
                "documentation_checked_utc": "2026-08-24T00:00:00Z",
                "streaming": True,
            },
        },
        "routes": [
            {
                "id": "portable-chat",
                "model": "portable-model",
                "lifecycle": {
                    "stage": "ga",
                    "status": "current",
                    "released_at_utc": "2026-01-01T00:00:00Z",
                },
                "role": "primary",
                "live_admission": {
                    "status": "live_proved",
                    "verified_at_utc": "2026-08-25T00:00:00Z",
                    "evidence_sha256": "b" * 64,
                },
            }
        ],
    }


def _experiment_profile() -> dict[str, Any]:
    return {
        "schema": "benchmark-experiment/v1",
        "as_of_utc": AS_OF_UTC,
        "campaign": {
            "name": "portable-v2-contract-test",
            "seed": 23,
            "max_wall_seconds": 600,
            "max_cost_usd": 20,
            "launch_reserve_seconds": 30,
            "launch_reserve_usd": 1,
            "concurrency": 4,
            "retries": 0,
            "client_location": "portable-client-region",
        },
        "suites": {"static": {"enabled": True, "offered_rps": 1}},
    }


def _route(profile: dict[str, Any]) -> dict[str, Any]:
    return profile["routes"][0]


def test_v2_admits_current_ga_route() -> None:
    compilation = compose_profile_config(_provider_profile_v2(), _experiment_profile())

    assert [route.id for route in compilation.config.routes] == ["portable-chat"]
    assert "lifecycle" not in compilation.mapping["routes"][0]
    assert "live_admission" not in compilation.mapping["routes"][0]


def test_v2_admits_current_preview_lane() -> None:
    provider = _provider_profile_v2()
    route = _route(provider)
    route["lifecycle"]["stage"] = "preview"
    route["role"] = "preview"

    compilation = compose_profile_config(provider, _experiment_profile())

    assert compilation.config.routes[0].id == "portable-chat"


def test_v2_rejects_stale_catalog_at_experiment_as_of() -> None:
    provider = _provider_profile_v2()
    provider["catalog"]["documentation_checked_at_utc"] = "2026-06-30T00:00:00Z"
    provider["catalog"]["revalidated_at_utc"] = "2026-07-01T00:00:00Z"

    with pytest.raises(ValueError, match="stale catalog"):
        compose_profile_config(provider, _experiment_profile())


def test_v2_admits_route_with_future_retirement() -> None:
    provider = _provider_profile_v2()
    _route(provider)["lifecycle"]["retirement_at_utc"] = "2026-09-30T00:00:00Z"

    compilation = compose_profile_config(provider, _experiment_profile())

    assert compilation.config.routes[0].id == "portable-chat"


def test_v2_rejects_route_with_past_retirement() -> None:
    provider = _provider_profile_v2()
    _route(provider)["lifecycle"]["retirement_at_utc"] = "2026-08-29T00:00:00Z"

    with pytest.raises(ValueError, match="retired as of"):
        compose_profile_config(provider, _experiment_profile())


def test_v2_rejects_superseded_selected_route() -> None:
    provider = _provider_profile_v2()
    lifecycle = _route(provider)["lifecycle"]
    lifecycle["status"] = "superseded"
    lifecycle["superseded_by"] = "portable-chat-next"

    with pytest.raises(ValueError, match="superseded by"):
        compose_profile_config(provider, _experiment_profile())


def test_v2_rejects_explicitly_excluded_control_route() -> None:
    provider = _provider_profile_v2()
    route = _route(provider)
    route["role"] = "control"
    route["live_admission"] = {
        "status": "excluded",
        "reason": "historical control only",
    }

    with pytest.raises(ValueError, match="explicitly excluded"):
        compose_profile_config(provider, _experiment_profile())


def test_v2_rejects_unverified_live_route() -> None:
    provider = _provider_profile_v2()
    _route(provider)["live_admission"] = {
        "status": "unverified",
        "reason": "exact route canary has not run",
    }

    with pytest.raises(ValueError, match="not live-proved"):
        compose_profile_config(provider, _experiment_profile())


def test_v2_requires_deterministic_utc_as_of() -> None:
    experiment = copy.deepcopy(_experiment_profile())
    experiment.pop("as_of_utc")
    with pytest.raises(ValueError, match="as_of_utc is required"):
        compose_profile_config(_provider_profile_v2(), experiment)

    experiment["as_of_utc"] = "2026-08-30T01:00:00+01:00"
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        compose_profile_config(_provider_profile_v2(), experiment)
