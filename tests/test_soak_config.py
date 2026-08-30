from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from inference_bench.config import load_config
from inference_bench.soak_config import derive_soak_config


def _write_config(path: Path, route) -> Path:  # type: ignore[no-untyped-def]
    config = {
        "campaign": {
            "name": "derive-soak-test",
            "seed": 7,
            "max_wall_seconds": 600,
            "max_cost_usd": 100,
            "launch_reserve_seconds": 30,
            "launch_reserve_usd": 1,
            "concurrency": 4,
            "retries": 1,
            "client_location": "test-client",
        },
        "routes": [
            {
                "id": route.id,
                "provider": route.provider,
                "adapter": route.adapter,
                "model": route.model,
                "base_url": route.base_url,
                "auth": {"env": route.auth.env},
                "quota_scope": route.quota_scope,
                "region": route.region,
                "api_version": route.api_version,
                "model_version": route.model_version,
                "context_tokens": route.context_tokens,
                "max_output_tokens": route.max_output_tokens,
                "input_usd_per_million": route.input_usd_per_million,
                "output_usd_per_million": route.output_usd_per_million,
                "documentation_source_url": route.documentation_source_url,
                "pricing_source_url": route.pricing_source_url,
                "evidence_retrieved_at_utc": route.evidence_retrieved_at_utc,
                "evidence_bundle_sha256": route.evidence_bundle_sha256,
                "capabilities": route.capabilities,
            }
        ],
        "suites": {
            "aimd": {
                "enabled": True,
                "shapes": ["short_short", "long_short", "short_long", "mixed"],
                "baseline_samples": 20,
                "concurrency": 4,
            }
        },
    }
    target = path / "source.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _controller_csv(path: Path, route_id: str, *, healthy: str = "2.5") -> None:
    fields = [
        "suite",
        "route_id",
        "shape",
        "controller_completion_state",
        "capacity_bound_state",
        "healthy_lower_bound_rps",
        "confirmation_complete",
        "confirmation_all_healthy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shape in ("short_short", "long_short", "short_long", "mixed"):
            writer.writerow(
                {
                    "suite": "aimd",
                    "route_id": route_id,
                    "shape": shape,
                    "controller_completion_state": "completed_confirmations_healthy",
                    "capacity_bound_state": "bracketed_healthy_lower_unhealthy_upper",
                    "healthy_lower_bound_rps": healthy,
                    "confirmation_complete": "True",
                    "confirmation_all_healthy": "True",
                }
            )


def test_derive_soak_config_copies_observed_cell_rate(tmp_path: Path, route) -> None:
    source = _write_config(tmp_path, route)
    route_id = load_config(source).routes[0].id
    summary = tmp_path / "controller-summary.csv"
    _controller_csv(summary, route_id)
    output = tmp_path / "soak.yaml"

    derive_soak_config(source, summary, output)

    loaded = load_config(output)
    assert loaded.suites["aimd"]["enabled"] is False
    soak = loaded.suites["soak"]
    assert soak["blocks"] == 4
    assert soak["block_seconds"] == 30
    assert soak["rate_rps_by_route_shape"][route_id]["mixed"] == 2.5
    provenance = json.loads(output.with_suffix(".rates.json").read_text(encoding="utf-8"))
    assert {cell["basis"] for cell in provenance["cells"]} == {"observed_aimd_healthy_lower_bound"}
    assert provenance["source_campaign_identity_sha256"] == load_config(source).identity_hash
    assert len(provenance["source_config_sha256"]) == 64
    assert len(provenance["controller_summary_sha256"]) == 64
    assert provenance["route_profile_overrides"] == {}
    assert provenance["route_profile_overrides_sha256"] is None
    assert provenance["derived_campaign_identity_sha256"] == loaded.identity_hash


def test_derive_soak_requires_explicit_fallback_for_missing_bound(tmp_path: Path, route) -> None:
    source = _write_config(tmp_path, route)
    route_id = load_config(source).routes[0].id
    summary = tmp_path / "controller-summary.csv"
    _controller_csv(summary, route_id, healthy="")

    with pytest.raises(ValueError, match="explicit --fallback-rps"):
        derive_soak_config(source, summary, tmp_path / "blocked.yaml")

    output = tmp_path / "exploratory.yaml"
    derive_soak_config(source, summary, output, fallback_rps=0.1)
    raw = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert raw["suites"]["soak"]["rate_rps_by_route_shape"][route_id]["short_short"] == 0.1
    provenance = json.loads(output.with_suffix(".rates.json").read_text(encoding="utf-8"))
    assert {cell["basis"] for cell in provenance["cells"]} == {
        "explicit_exploratory_fallback_no_healthy_aimd_bound"
    }


def test_derive_soak_applies_typed_route_profile_with_identity_provenance(
    tmp_path: Path, route
) -> None:
    source = _write_config(tmp_path, route)
    source_config = load_config(source)
    route_id = source_config.routes[0].id
    summary = tmp_path / "controller-summary.csv"
    _controller_csv(summary, route_id)
    profile = tmp_path / "route-profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "schema": "inference-bench-route-profile-overrides/v1",
                "routes": {
                    route_id: {
                        "output_limit_scope": "visible_text",
                        "output_limit_tolerance_tokens": 32,
                        "reasoning_reservation_tokens": 4096,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "soak-profiled.yaml"

    derive_soak_config(source, summary, output, route_profile_overrides=profile)

    loaded = load_config(output)
    configured = loaded.routes[0]
    assert configured.output_limit_scope == "visible_text"
    assert configured.output_limit_tolerance_tokens == 32
    assert configured.reasoning_reservation_tokens == 4096
    provenance = json.loads(output.with_suffix(".rates.json").read_text(encoding="utf-8"))
    assert provenance["source_campaign_identity_sha256"] == source_config.identity_hash
    assert provenance["derived_campaign_identity_sha256"] == loaded.identity_hash
    assert provenance["derived_campaign_identity_sha256"] != source_config.identity_hash
    assert provenance["route_profile_overrides"][route_id]["output_limit_scope"] == "visible_text"
    assert len(provenance["route_profile_overrides_sha256"]) == 64


def test_derive_soak_rejects_unsupported_route_profile_field(tmp_path: Path, route) -> None:
    source = _write_config(tmp_path, route)
    route_id = load_config(source).routes[0].id
    summary = tmp_path / "controller-summary.csv"
    _controller_csv(summary, route_id)
    profile = tmp_path / "route-profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "schema": "inference-bench-route-profile-overrides/v1",
                "routes": {route_id: {"base_url": "https://example.invalid"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields: base_url"):
        derive_soak_config(
            source,
            summary,
            tmp_path / "blocked.yaml",
            route_profile_overrides=profile,
        )
