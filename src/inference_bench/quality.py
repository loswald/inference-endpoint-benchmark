from __future__ import annotations

from typing import Any

from .json_contract import StrictJSONError, strict_json_loads
from .models import InferenceResult, RequestSpec

SUPPORTED_SCORERS = frozenset(
    {
        "exact",
        "contains",
        "json_fields",
        "json_answer_7",
        "tool_city_reykjavik",
        "tool_cities_reykjavik_tokyo",
        "tool_nested_trip",
        "tool_named_target",
        "exact_blue",
        "exact_red_blue",
        "context_markers",
        "json_keys",
        "longform_completion",
    }
)


def predeclared_quality_scorer(spec: RequestSpec) -> str | None:
    """Return the immutable deterministic scorer selected before a request is claimed."""

    scorer = spec.metadata.get("scorer") or spec.metadata.get("quality")
    return scorer if isinstance(scorer, str) and scorer in SUPPORTED_SCORERS else None


def _strict_json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = strict_json_loads(value)
    except StrictJSONError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _type_exact_equal(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _type_exact_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _type_exact_equal(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return bool(observed == expected)


def score_result(spec: RequestSpec, result: InferenceResult) -> tuple[float | None, dict[str, Any]]:
    scorer = predeclared_quality_scorer(spec)
    expected = spec.metadata.get("expected")
    if not scorer:
        return None, {"scorer": "none"}
    if result.status != "success":
        return 0.0, {
            "scorer": str(scorer),
            "deterministic": True,
            "quality_estimand": "end_to_end_all_predeclared_trials_non_success_is_zero",
        }
    text = result.output_text.strip()
    if scorer == "exact":
        score = float(text == str(expected))
    elif scorer == "contains":
        score = float(str(expected) in text)
    elif scorer == "json_fields":
        parsed = _strict_json_object(text)
        expected_object = expected if isinstance(expected, dict) else None
        score = float(
            parsed is not None
            and expected_object is not None
            and _type_exact_equal(parsed, expected_object)
        )
    elif scorer == "json_answer_7":
        parsed = _strict_json_object(text)
        score = float(parsed is not None and _type_exact_equal(parsed, {"answer": 7}))
    elif scorer == "tool_city_reykjavik":
        matches = []
        malformed = 0
        for call in result.tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get("name") != "lookup_weather":
                malformed += 1
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                malformed += 1
                continue
            parsed_arguments = _strict_json_object(arguments)
            if parsed_arguments is None:
                malformed += 1
                continue
            city = parsed_arguments.get("city")
            if set(parsed_arguments) == {"city"} and isinstance(city, str):
                matches.append(city.strip().casefold())
            else:
                malformed += 1
        score = float(matches == ["reykjavík".casefold()] and malformed == 0)
        return score, {
            "scorer": str(scorer),
            "deterministic": True,
            "matching_tool_calls": len(matches),
            "malformed_matching_calls": malformed,
        }
    elif scorer == "tool_cities_reykjavik_tokyo":
        cities: list[str] = []
        malformed = 0
        for call in result.tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            parsed_arguments = (
                _strict_json_object(arguments) if isinstance(arguments, str) else None
            )
            if (
                not isinstance(function, dict)
                or function.get("name") != "lookup_weather"
                or parsed_arguments is None
                or set(parsed_arguments) != {"city"}
                or not isinstance(parsed_arguments.get("city"), str)
            ):
                malformed += 1
                continue
            cities.append(parsed_arguments["city"].strip().casefold())
        score = float(
            sorted(cities) == sorted(["reykjavík".casefold(), "tokyo"]) and malformed == 0
        )
        return score, {
            "scorer": str(scorer),
            "deterministic": True,
            "matching_tool_calls": len(cities),
            "malformed_matching_calls": malformed,
        }
    elif scorer == "tool_nested_trip":
        expected_trip = {
            "destination": "Lisbon",
            "passenger": {"name": "Ada", "contact": {"email": "ada@example.com"}},
        }
        matches = []
        for call in result.tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            parsed_arguments = (
                _strict_json_object(arguments) if isinstance(arguments, str) else None
            )
            if isinstance(function, dict) and function.get("name") == "book_trip":
                matches.append(parsed_arguments)
        score = float(len(matches) == 1 and _type_exact_equal(matches[0], expected_trip))
    elif scorer == "tool_named_target":
        names = [
            call.get("function", {}).get("name")
            for call in result.tool_calls
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        ]
        score = float(names == [str(expected)])
    elif scorer == "exact_blue":
        score = float(text.casefold().rstrip(".") == "blue")
    elif scorer == "exact_red_blue":
        score = float(text.casefold().replace(" ", "").rstrip(".") == "red|blue")
    elif scorer == "context_markers":
        markers = spec.metadata.get("context_markers")
        score = float(
            isinstance(markers, list)
            and len(markers) == 3
            and all(isinstance(marker, str) for marker in markers)
            and text == "|".join(markers)
        )
    elif scorer == "json_keys":
        parsed = _strict_json_object(text)
        expected_keys = spec.metadata.get("expected_keys")
        score = float(
            parsed is not None
            and isinstance(expected_keys, list)
            and all(isinstance(key, str) for key in expected_keys)
            and set(parsed) == set(expected_keys)
        )
    elif scorer == "longform_completion":
        minimum_output_tokens = spec.metadata.get("minimum_output_tokens")
        score = float(
            isinstance(minimum_output_tokens, int)
            and minimum_output_tokens > 0
            and isinstance(result.output_tokens, int)
            and result.output_tokens >= minimum_output_tokens
            and bool(text)
        )
    else:
        return None, {"scorer": "none"}
    return score, {"scorer": str(scorer), "deterministic": True}
