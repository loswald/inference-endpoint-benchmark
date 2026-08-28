#!/usr/bin/env python3
"""Build the plain-language DigitalOcean interim report without rendering a PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference_bench.digitalocean_atlas import _build_interim_markdown, _read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capacity-source", required=True)
    parser.add_argument("--fixed-rate-source", required=True)
    parser.add_argument("--exclude-endpoint", action="append", default=[])
    args = parser.parse_args()

    source = Path(args.summary_dir).resolve()
    output = Path(args.output).resolve()
    excluded = set(args.exclude_endpoint)

    def included(name: str) -> list[dict[str, str]]:
        return [
            row
            for row in _read_csv(source / name)
            if str(row.get("endpoint_id") or "") not in excluded
        ]

    report = _build_interim_markdown(
        included("endpoint-inventory.csv"),
        included("capacity-summary.csv"),
        included("soak-cell-summary.csv"),
        included("coverage-matrix.csv"),
        included("soak-block-summary.csv"),
        included("recovery-summary.csv"),
        capacity_source=args.capacity_source,
        fixed_rate_source=args.fixed_rate_source,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
