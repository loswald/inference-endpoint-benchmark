# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from inference_bench.publication_safety import scan_publication


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed on unsafe public benchmark artifacts."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = scan_publication(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
