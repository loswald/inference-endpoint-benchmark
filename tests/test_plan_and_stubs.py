from __future__ import annotations

from inference_bench.adapters.base import FailClosedAdapter, adapter_for
from inference_bench.config import CampaignConfig
from inference_bench.models import AuthConfig, RouteConfig
from inference_bench.plan import build_plan


def test_native_stub_is_honestly_labelled(route) -> None:
    stub = RouteConfig(
        id="bedrock",
        provider="amazon-bedrock",
        adapter="bedrock_native",
        model="exact-model",
        base_url="native-sdk",
        auth=AuthConfig(env="AWS_ACCESS_KEY_ID"),
        input_usd_per_million=1,
        output_usd_per_million=1,
    )
    config = CampaignConfig(
        name="stub",
        seed=1,
        max_wall_seconds=100,
        max_cost_usd=10,
        launch_reserve_seconds=10,
        launch_reserve_usd=1,
        concurrency=1,
        retries=0,
        routes=(stub,),
        suites={"latency": {"enabled": True, "repeats": 1, "shapes": ["short_short"]}},
    )
    assert build_plan(config).native_placeholder_routes == ("bedrock",)
    assert isinstance(adapter_for("bedrock_native"), FailClosedAdapter)
