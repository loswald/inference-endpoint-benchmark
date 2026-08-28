from pathlib import Path

from inference_bench.cli import _apply_live_overrides, _parser
from inference_bench.config import load_config


def test_run_overrides_apply_suite_wall_and_cost() -> None:
    config_path = Path(__file__).parents[1] / "examples" / "digitalocean-hosted-2026-08-27.yaml"
    args = _parser().parse_args(
        [
            "run",
            str(config_path),
            "--only-suite",
            "aimd",
            "--max-wall-seconds",
            "6600",
            "--max-cost-usd",
            "338",
            "--output",
            "ignored",
            "--confirm-live",
        ]
    )

    amended = _apply_live_overrides(load_config(config_path), args)

    assert set(amended.suites) == {"aimd"}
    assert amended.max_wall_seconds == 6600
    assert amended.max_cost_usd == 338
