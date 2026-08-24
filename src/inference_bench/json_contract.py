from __future__ import annotations

import json
from typing import Any


class StrictJSONError(ValueError):
    """Raised when input is not finite, duplicate-free standards-compliant JSON."""


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-standard JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Parse finite JSON while rejecting duplicate object names at every nesting level."""

    try:
        return json.loads(
            value,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise StrictJSONError("invalid JSON") from exc
