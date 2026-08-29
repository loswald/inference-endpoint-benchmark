#!/usr/bin/env python3
"""Build a compact, public verification layer from a terminal harness report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _integer(row: dict[str, str], field: str) -> int:
    value = row.get(field, "").strip()
    return int(value) if value else 0


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "").strip()
    return float(value) if value else 0.0


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    source: Path,
    destination: Path,
    campaign_id: str,
    bundle_sha256: str,
    excluded_capacity_bundle_sha256: str | None,
) -> None:
    matched_path = source / "matched-cell-summary.csv"
    coverage_path = source / "coverage-ledger.csv"
    audit_path = source / "outlier-audit-summary.csv"
    matched = _read_csv(matched_path)
    coverage = _read_csv(coverage_path)
    audit = _read_csv(audit_path)
    if any("arcee" in (row.get("route_id") or "").lower() for row in matched + coverage):
        raise ValueError("Arcee rows are not allowed in the hosted-only verification layer")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in matched:
        grouped[(row["route_id"], row["suite"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for (endpoint_id, suite), rows in sorted(grouped.items()):
        attempts = sum(_integer(row, "attempts_n") for row in rows)
        successes = sum(_integer(row, "successes_n") for row in rows)
        summary_rows.append(
            {
                "campaign_id": campaign_id,
                "endpoint_id": endpoint_id,
                "suite": suite,
                "matched_cells_n": len(rows),
                "logical_requests_n": sum(_integer(row, "logical_requests_n") for row in rows),
                "attempts_n": attempts,
                "successes_n": successes,
                "success_rate": "" if not attempts else f"{successes / attempts:.12g}",
                "rate_limited_n": sum(_integer(row, "physical_rate_limited_n") for row in rows),
                "timeouts_n": sum(_integer(row, "physical_timeouts_n") for row in rows),
                "server_errors_n": sum(_integer(row, "physical_server_errors_n") for row in rows),
                "unexpected_client_errors_n": sum(
                    _integer(row, "unexpected_client_error_n") for row in rows
                ),
                "cache_hit_n": sum(_integer(row, "cache_hit_n") for row in rows),
                "cache_miss_n": sum(_integer(row, "cache_miss_n") for row in rows),
                "cache_unknown_n": sum(_integer(row, "cache_read_unknown_n") for row in rows),
                "cache_read_tokens_sum": sum(
                    _integer(row, "cache_read_tokens_sum") for row in rows
                ),
                "settled_usd_sum": f"{sum(_number(row, 'settled_usd_sum') for row in rows):.12g}",
                "source_bundle_sha256": bundle_sha256,
            }
        )

    cache_cells = {
        (row["route_id"], row["cache_state"]): row
        for row in matched
        if row.get("suite") == "cache"
    }
    cache_rows: list[dict[str, object]] = []
    endpoints = sorted({endpoint for endpoint, _ in cache_cells})
    for endpoint_id in endpoints:
        cached = cache_cells.get((endpoint_id, "cached_trial"), {})
        uncached = cache_cells.get((endpoint_id, "uncached_trial"), {})
        cached_cost = _number(cached, "settled_usd_sum")
        uncached_cost = _number(uncached, "settled_usd_sum")
        cache_rows.append(
            {
                "campaign_id": campaign_id,
                "endpoint_id": endpoint_id,
                "cached_requests_n": _integer(cached, "attempts_n"),
                "uncached_requests_n": _integer(uncached, "attempts_n"),
                "cached_token_hits_n": _integer(cached, "cache_hit_n"),
                "cached_token_misses_n": _integer(cached, "cache_miss_n"),
                "cache_read_tokens_sum": _integer(cached, "cache_read_tokens_sum"),
                "cached_ttft_p50_seconds": cached.get("ttft_p50", ""),
                "cached_ttft_p50_ci95_low": cached.get("ttft_p50_ci95_low", ""),
                "cached_ttft_p50_ci95_high": cached.get("ttft_p50_ci95_high", ""),
                "uncached_ttft_p50_seconds": uncached.get("ttft_p50", ""),
                "uncached_ttft_p50_ci95_low": uncached.get("ttft_p50_ci95_low", ""),
                "uncached_ttft_p50_ci95_high": uncached.get("ttft_p50_ci95_high", ""),
                "cached_settled_usd": f"{cached_cost:.12g}",
                "uncached_settled_usd": f"{uncached_cost:.12g}",
                "observed_cost_ratio_cached_over_uncached": (
                    "" if not uncached_cost else f"{cached_cost / uncached_cost:.12g}"
                ),
                "source_bundle_sha256": bundle_sha256,
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(
        destination / "static-verification-summary.csv",
        summary_rows,
        list(summary_rows[0]) if summary_rows else [],
    )
    _write_csv(
        destination / "cache-verification-pairs.csv",
        cache_rows,
        list(cache_rows[0]) if cache_rows else [],
    )
    coverage_counts = Counter(row.get("state") or "unknown" for row in coverage)
    manifest = {
        "schema_version": "digitalocean-static-verification/v1",
        "campaign_id": campaign_id,
        "source_bundle_sha256": bundle_sha256,
        "source_files": {
            path.name: _sha256(path) for path in (matched_path, coverage_path, audit_path)
        },
        "matched_cells": len(matched),
        "endpoint_count_with_completed_cells": len({row["route_id"] for row in matched}),
        "coverage_states": dict(sorted(coverage_counts.items())),
        "outlier_audit_counts": {
            row.get("audit_class", "unknown"): int(row.get("n") or 0) for row in audit
        },
        "capacity_evidence_contributed": False,
        "capacity_exclusion_reason": "AIMD cells were campaign-censored before start",
        "excluded_capacity_closure_bundle_sha256": excluded_capacity_bundle_sha256,
        "excluded_capacity_closure_reason": (
            "live CLI overrides were accepted but not applied; static suites repeated and all "
            "AIMD cells remained campaign-censored before start"
            if excluded_capacity_bundle_sha256
            else None
        ),
    }
    (destination / "static-verification-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--excluded-capacity-bundle-sha256")
    args = parser.parse_args()
    build(
        args.source,
        args.destination,
        args.campaign_id,
        args.bundle_sha256,
        args.excluded_capacity_bundle_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
