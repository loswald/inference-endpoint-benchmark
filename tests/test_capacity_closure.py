from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from inference_bench.capacity_closure import (
    build_capacity_closure_package,
    build_capacity_closure_package_from_files,
    capacity_workload_identity,
    export_controller_capacity_evidence,
    validate_capacity_closure_profile,
)
from inference_bench.cli import main
from inference_bench.config import CampaignConfig, load_config, selected_capacity_cells
from inference_bench.models import AuthConfig, RouteConfig

EVIDENCE_SHA = "a" * 64
IDENTITY_FIELDS = (
    "route_identity_sha256",
    "source_campaign_identity_sha256",
    "workload_recipe_sha256",
    "input_target",
    "output_target",
)


def _route(
    route_id: str,
    provider: str,
    *,
    upstream_provider: str | None = None,
) -> RouteConfig:
    return RouteConfig(
        id=route_id,
        provider=provider,
        adapter="openai_compatible",
        model=f"{route_id}-model",
        base_url=f"https://{provider}.example.test/v1/chat/completions",
        auth=AuthConfig(env=f"{provider.upper()}_API_KEY"),
        region="test-region-1",
        api_family="chat_completions",
        api_version="v1",
        model_version="fixture-v1",
        upstream_provider=upstream_provider,
        quota_scope="fixture-account",
        context_tokens=16_384,
        max_output_tokens=4_096,
        request_timeout_seconds=30,
        transport_max_connections=16,
        input_usd_per_million=0.1,
        output_usd_per_million=0.2,
        documentation_source_url=f"https://{provider}.example.test/docs",
        pricing_source_url=f"https://{provider}.example.test/pricing",
        evidence_retrieved_at_utc="2030-01-01T00:00:00Z",
        evidence_bundle_sha256=EVIDENCE_SHA,
        capabilities={"documentation_checked_utc": "2030-01-01T00:00:00Z"},
    )


def _base_config() -> CampaignConfig:
    return CampaignConfig(
        name="portable-fixture",
        seed=7,
        max_wall_seconds=3_600,
        max_cost_usd=50,
        launch_reserve_seconds=60,
        launch_reserve_usd=1,
        concurrency=16,
        retries=1,
        client_location="test-client-region",
        routes=(
            _route("nebula-a", "nebula"),
            _route("quasar-b", "quasar"),
            _route("relay-c", "nebula", upstream_provider="relay-network"),
        ),
        suites={
            "static": {"enabled": True, "offered_rps": 1},
            "aimd": {
                "enabled": True,
                "shapes": ["short_short", "long_short"],
                "initial_rps": 0.25,
                "additive_rps": 0.25,
                "multiplicative_decrease": 0.5,
                "bracket_epochs": 2,
                "bracket_multiplier": 2,
                "max_rps": 4,
                "epochs": 4,
                "epoch_seconds": 5,
                "concurrency": 16,
                "baseline_rps": 0.25,
                "baseline_samples": 20,
                "baseline_attempts": 3,
                "baseline_multiplicative_decrease": 0.5,
                "confirmation_max_stages": 4,
                "confirmation_multiplicative_decrease": 0.5,
                "confirmation_separator_samples": 5,
                "minimum_rps": 0.0625,
                "long_input_tokens": 4_096,
                "long_input_overflow": "fail",
                "long_output_tokens": 128,
                "long_output_overflow": "fail",
            },
        },
    )


def _base_config_mapping() -> dict[str, object]:
    config = _base_config()
    return {
        "campaign": {
            field: getattr(config, field)
            for field in (
                "name",
                "seed",
                "max_wall_seconds",
                "max_cost_usd",
                "launch_reserve_seconds",
                "launch_reserve_usd",
                "concurrency",
                "retries",
                "input_token_reservation_factor",
                "client_location",
            )
        },
        "routes": [
            {
                **asdict(route),
                "retained_header_names": list(route.retained_header_names),
            }
            for route in config.routes
        ],
        "suites": config.suites,
    }


def _profile(*, expected_cells: int = 2) -> dict[str, object]:
    return {
        "schema": "capacity-closure-profile/v1",
        "suite": "aimd",
        "selection": {
            "columns": {
                "route_id": "route_key",
                "workload": "workload_key",
                "prior_state": "evidence_state",
                "route_identity_sha256": "route_identity_sha256",
                "source_campaign_identity_sha256": "source_campaign_identity_sha256",
                "workload_recipe_sha256": "workload_recipe_sha256",
                "input_target": "input_target",
                "output_target": "output_target",
            },
            "where": [
                {"field": "experiment", "op": "equals", "value": "closure-candidate"},
                {"field": "evidence_state", "op": "in", "value": ["floor-unresolved"]},
            ],
            "expected_cells": expected_cells,
        },
        "mapping": {
            "workload_to_shape": {
                "brief": "short_short",
                "prompt-heavy": "long_short",
            }
        },
        "route_predicates": [
            {"field": "provider", "op": "in", "value": ["nebula", "quasar"]},
            {"field": "adapter", "op": "equals", "value": "openai_compatible"},
            {"field": "upstream_provider", "op": "empty"},
        ],
        "overrides": {
            "campaign": {
                "name": "portable-fixture-closure",
                "seed": 11,
                "max_wall_seconds": 7_200,
                "max_cost_usd": 100,
                "launch_reserve_seconds": 120,
                "launch_reserve_usd": 2,
                "concurrency": 16,
                "retries": 0,
            },
            "suite": {
                "baseline_rps": 0.25,
                "baseline_attempts": 4,
                "minimum_rps": 0.03125,
                "confirmation_max_stages": 8,
            },
        },
        "output": {
            "config_filename": "capacity-closure.yaml",
            "manifest_filename": "plan.json",
        },
    }


def _write_evidence(
    path: Path,
    rows: list[dict[str, str]],
    *,
    source_config: CampaignConfig | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "route_key",
                "workload_key",
                "evidence_state",
                *IDENTITY_FIELDS,
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **_identity_fields(
                        row["route_key"], row["workload_key"], source_config=source_config
                    ),
                    **row,
                }
            )


def _identity_fields(
    route_id: str,
    workload: str,
    *,
    source_config: CampaignConfig | None = None,
) -> dict[str, str]:
    config = source_config or _base_config()
    route = next(route for route in config.routes if route.id == route_id)
    shape = {"brief": "short_short", "prompt-heavy": "long_short"}[workload]
    identity = capacity_workload_identity(route, "aimd", config.suites["aimd"], shape)
    return {
        **identity,
        "source_campaign_identity_sha256": config.identity_hash,
    }


def _valid_rows() -> list[dict[str, str]]:
    return [
        {
            "experiment": "closure-candidate",
            "route_key": "nebula-a",
            "workload_key": "brief",
            "evidence_state": "floor-unresolved",
        },
        {
            "experiment": "closure-candidate",
            "route_key": "quasar-b",
            "workload_key": "prompt-heavy",
            "evidence_state": "floor-unresolved",
        },
        {
            "experiment": "historical",
            "route_key": "relay-c",
            "workload_key": "brief",
            "evidence_state": "confirmed",
        },
    ]


def test_provider_neutral_closure_uses_evidence_mapping_and_exact_cells(tmp_path: Path) -> None:
    evidence = tmp_path / "capacity.csv"
    _write_evidence(evidence, _valid_rows())

    config_path, manifest_path = build_capacity_closure_package(
        _base_config(), evidence, _profile(), tmp_path / "package"
    )

    config = load_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(config.suites) == {"aimd"}
    assert [route.id for route in config.routes] == ["nebula-a", "quasar-b"]
    assert {(route.id, shape) for route, shape in selected_capacity_cells(config, "aimd")} == {
        ("nebula-a", "short_short"),
        ("quasar-b", "long_short"),
    }
    assert config.retries == 0
    assert config.suites["aimd"]["minimum_rps"] == 0.03125
    assert manifest["schema"] == "capacity-closure-plan/v1"
    assert manifest["selection"]["cell_count"] == 2
    assert manifest["selection"]["route_count"] == 2
    assert manifest["execution_contract"]["suite_name"] == "aimd"
    assert manifest["execution_contract"]["excluded_base_suites"] == ["static"]
    assert manifest["live_traffic_sent"] is False
    assert manifest["plan"]["static_requests"] == 0

    combined = config_path.read_text(encoding="utf-8") + manifest_path.read_text(
        encoding="utf-8"
    )
    assert "Bearer fixture-secret" not in combined
    assert "relay-network" not in combined


def test_controller_summary_export_is_terminal_identity_bound_and_secret_free(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(_base_config_mapping(), sort_keys=False), encoding="utf-8")
    loaded = load_config(source)
    summary = tmp_path / "controller-summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "suite",
                "route_id",
                "shape",
                "controller_completion_state",
                "capacity_bound_state",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "suite": "aimd",
                "route_id": "nebula-a",
                "shape": "short_short",
                "controller_completion_state": "campaign_guard_censored",
                "capacity_bound_state": "campaign_guard_censored_before_confirmation",
            }
        )
    report = tmp_path / "reproducibility-manifest.json"
    report.write_text(
        json.dumps(
            {
                "campaign": {
                    "identity_hash": loaded.identity_hash,
                    "ended_at_utc": "2030-01-01T01:00:00Z",
                    "terminal_event": {"reason": "budget_guard"},
                }
            }
        ),
        encoding="utf-8",
    )

    output, manifest_path = export_controller_capacity_evidence(
        source, summary, report, tmp_path / "bound.csv"
    )

    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert rows[0]["source_campaign_identity_sha256"] == loaded.identity_hash
    assert len(rows[0]["route_identity_sha256"]) == 64
    assert len(rows[0]["workload_recipe_sha256"]) == 64
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "capacity-controller-evidence/v1"
    assert manifest["cell_count"] == 1
    assert manifest["live_traffic_sent"] is False
    combined = output.read_text(encoding="utf-8") + manifest_path.read_text(encoding="utf-8")
    assert "Bearer fixture-secret" not in combined
    assert "https://" not in combined


def test_controller_summary_export_rejects_cross_campaign_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(_base_config_mapping(), sort_keys=False), encoding="utf-8")
    summary = tmp_path / "controller-summary.csv"
    summary.write_text(
        "suite,route_id,shape,controller_completion_state\n"
        "aimd,nebula-a,short_short,campaign_guard_censored\n",
        encoding="utf-8",
    )
    report = tmp_path / "reproducibility-manifest.json"
    report.write_text(
        json.dumps(
            {
                "campaign": {
                    "identity_hash": "0" * 64,
                    "ended_at_utc": "2030-01-01T01:00:00Z",
                    "terminal_event": {"reason": "budget_guard"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        export_controller_capacity_evidence(source, summary, report, tmp_path / "blocked.csv")


def test_capacity_closure_rejects_selected_pass_through_route(tmp_path: Path) -> None:
    evidence = tmp_path / "capacity.csv"
    _write_evidence(
        evidence,
        [
            {
                "experiment": "closure-candidate",
                "route_key": "relay-c",
                "workload_key": "brief",
                "evidence_state": "floor-unresolved",
            }
        ],
    )
    profile = _profile(expected_cells=1)

    with pytest.raises(ValueError, match="violates route_predicates: upstream_provider empty"):
        build_capacity_closure_package(_base_config(), evidence, profile, tmp_path / "package")


def test_capacity_closure_profile_requires_explicit_provider_but_not_direct_routing() -> None:
    profile = _profile()
    profile["route_predicates"] = [
        {"field": "adapter", "op": "equals", "value": "openai_compatible"}
    ]
    with pytest.raises(ValueError, match="explicitly select one or more providers"):
        validate_capacity_closure_profile(profile)

    profile = _profile()
    profile["route_predicates"] = [
        {"field": "provider", "op": "in", "value": ["nebula", "quasar"]}
    ]
    assert validate_capacity_closure_profile(profile)["route_predicates"] == profile[
        "route_predicates"
    ]


def test_capacity_closure_can_explicitly_select_a_pinned_pass_through_route(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "capacity.csv"
    _write_evidence(
        evidence,
        [
            {
                "experiment": "closure-candidate",
                "route_key": "relay-c",
                "workload_key": "brief",
                "evidence_state": "floor-unresolved",
            }
        ],
    )
    profile = _profile(expected_cells=1)
    profile["route_predicates"] = [
        {"field": "provider", "op": "equals", "value": "nebula"},
        {"field": "upstream_provider", "op": "equals", "value": "relay-network"},
    ]

    config_path, _manifest_path = build_capacity_closure_package(
        _base_config(), evidence, profile, tmp_path / "package"
    )
    assert [route.id for route in load_config(config_path).routes] == ["relay-c"]


@pytest.mark.parametrize("identity_field", IDENTITY_FIELDS)
def test_capacity_closure_rejects_each_mismatched_source_identity(
    tmp_path: Path, identity_field: str
) -> None:
    evidence = tmp_path / f"capacity-{identity_field}.csv"
    rows = _valid_rows()
    rows[0][identity_field] = "mismatch"
    _write_evidence(evidence, rows)

    with pytest.raises(ValueError, match="capacity evidence identity does not match"):
        build_capacity_closure_package(_base_config(), evidence, _profile(), tmp_path / "package")


def test_capacity_closure_file_wrapper_cli_and_output_are_byte_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_path = tmp_path / "base.yaml"
    base_document = _base_config_mapping()
    base_path.write_text(yaml.safe_dump(base_document, sort_keys=False), encoding="utf-8")
    evidence = tmp_path / "capacity.csv"
    _write_evidence(evidence, _valid_rows(), source_config=load_config(base_path))
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(_profile(), sort_keys=False), encoding="utf-8")

    first = build_capacity_closure_package_from_files(
        base_path, evidence, profile_path, tmp_path / "first"
    )
    second = build_capacity_closure_package_from_files(
        base_path, evidence, profile_path, tmp_path / "second"
    )
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()

    cli_output = tmp_path / "cli"
    assert (
        main(
            [
                "plan-capacity-closure",
                str(base_path),
                str(evidence),
                str(profile_path),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    emitted = json.loads(capsys.readouterr().out)
    assert Path(emitted["config"]).is_file()
    assert Path(emitted["plan"]).is_file()
