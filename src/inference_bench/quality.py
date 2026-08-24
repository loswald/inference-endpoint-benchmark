from __future__ import annotations

import json
from typing import Any

from .models import InferenceResult, RequestSpec


def score_result(spec: RequestSpec, result: InferenceResult) -> tuple[float | None, dict[str, Any]]:
    if result.status != "success":
        return None, {"scorer": "not_scored_non_success"}
    scorer = spec.metadata.get("scorer") or spec.metadata.get("quality")
    expected = spec.metadata.get("expected")
    text = result.output_text.strip()
    if scorer == "exact":
        score = float(text == str(expected))
    elif scorer == "contains":
        score = float(str(expected) in text)
    elif scorer == "json_fields":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            score = 0.0
        else:
            score = float(all(parsed.get(key) == value for key, value in dict(expected).items()))
    elif scorer == "json_answer_7":
        try:
            score = float(json.loads(text).get("answer") == 7)
        except (json.JSONDecodeError, AttributeError):
            score = 0.0
    elif scorer == "tool_city_reykjavik":
        serialized = json.dumps(result.tool_calls, ensure_ascii=False).lower()
        score = float("lookup_weather" in serialized and "reyk" in serialized)
    elif scorer == "exact_blue":
        score = float(text.casefold().rstrip(".") == "blue")
    elif scorer == "context_markers":
        nonce = str(spec.metadata["nonce"])
        score = float(all(f"{nonce}-{letter}" in text for letter in "ABC"))
    else:
        return None, {"scorer": "none"}
    return score, {"scorer": str(scorer), "deterministic": True}
