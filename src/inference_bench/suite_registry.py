from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from .models import RequestSpec

SuiteValidator = Callable[[dict[str, Any]], None]
SUITE_ENTRY_POINT_GROUP = "inference_endpoint_benchmark.suites"


@dataclass(frozen=True, slots=True)
class SuitePlugin:
    """A versioned, independently registrable request-planning suite."""

    id: str
    version: str
    planner: Callable[..., list[RequestSpec]]
    public_keys: frozenset[str]
    validator: SuiteValidator | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("suite plugin id must contain only letters, digits, '_' or '-'")
        if not self.version:
            raise ValueError("suite plugin version is required")
        if not callable(self.planner):
            raise TypeError("suite plugin planner must be callable")
        if self.validator is not None and not callable(self.validator):
            raise TypeError("suite plugin validator must be callable")


_SUITES: dict[str, SuitePlugin] = {}
_BUILTINS_REGISTERED = False
_ENTRY_POINTS_LOADED = False


def register_suite(plugin: SuitePlugin, *, replace: bool = False) -> None:
    if plugin.id in _SUITES and not replace:
        raise ValueError(f"suite already registered: {plugin.id}")
    _SUITES[plugin.id] = plugin


def _register_builtins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .workloads import PLANNERS

    for name, planner in PLANNERS.items():
        register_suite(
            SuitePlugin(
                id=name,
                version="builtin/v1",
                planner=planner,
                public_keys=frozenset(),
            )
        )
    _BUILTINS_REGISTERED = True


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    discovered = metadata.entry_points()
    selected = (
        discovered.select(group=SUITE_ENTRY_POINT_GROUP)
        if hasattr(discovered, "select")
        else discovered.get(SUITE_ENTRY_POINT_GROUP, ())
    )
    for entry_point in selected:
        loaded = entry_point.load()
        plugin = loaded() if callable(loaded) and not isinstance(loaded, SuitePlugin) else loaded
        if not isinstance(plugin, SuitePlugin):
            raise TypeError(f"suite entry point {entry_point.name!r} did not return SuitePlugin")
        if plugin.id != entry_point.name:
            raise ValueError(
                f"suite entry point {entry_point.name!r} returned plugin id {plugin.id!r}"
            )
        register_suite(plugin)
    _ENTRY_POINTS_LOADED = True


def suite_plugins() -> dict[str, SuitePlugin]:
    _register_builtins()
    _load_entry_points()
    return dict(_SUITES)


def available_suites() -> tuple[str, ...]:
    return tuple(sorted(suite_plugins()))


def suite_applies_to_route(values: dict[str, Any], route_id: str) -> bool:
    selected = values.get("route_ids")
    return selected is None or route_id in selected

