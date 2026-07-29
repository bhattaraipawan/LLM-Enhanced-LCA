"""Small, dependency-free helpers for imperfect model JSON."""

from __future__ import annotations

import json
from typing import Any


def extract_json_block(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    depth = 0
    start: int | None = None
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : index + 1]
    return text


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(extract_json_block(text))
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
