from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_bench.digitalocean_final import generate_digitalocean_final_report
from inference_bench.digitalocean_variation import build_variation_tables
from inference_bench.publication_manifest import build_publication_manifest
from inference_bench.publication_safety import scan_publication


def _write_safety_receipt(output: Path) -> None:
    receipt = output / "public-safety-scan.json"
    for _ in range(2):
        result = scan_publication(output)
        if not result["passed"]:
            raise ValueError("public artifact safety scan failed")
        receipt.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    build_publication_manifest(output)
    final_result = scan_publication(output)
    if not final_result["passed"]:
        raise ValueError("public artifact safety scan failed after manifest generation")
    final_rendered = json.dumps(final_result, indent=2, sort_keys=True) + "\n"
    if receipt.read_text(encoding="utf-8") != final_rendered:
        receipt.write_text(final_rendered, encoding="utf-8", newline="\n")
        build_publication_manifest(output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the identifier-free six-hour variation tables and final DigitalOcean PDF."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    build_variation_tables(args.run_dir, args.summary_dir, seed=args.seed)
    pdf = generate_digitalocean_final_report(
        args.summary_dir,
        args.summary_dir,
        args.output,
    )
    build_publication_manifest(args.output)
    _write_safety_receipt(args.output.resolve())
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
