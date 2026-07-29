"""Text normalization shared by process matching and workbook parsing."""

from __future__ import annotations

import re


def normalize_process_name(name: str) -> str:
    return " ".join(str(name or "").strip().rstrip(" ,.;").lower().split())


def search_tokens(text: str) -> list[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "at",
        "for",
        "from",
        "in",
        "market",
        "of",
        "on",
        "production",
        "the",
        "to",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalize_process_name(text))
        if len(token) > 1 and token not in stop_words
    ]


def material_name_from_query(material_query: str) -> str:
    query = " ".join(material_query.strip().split())
    normalized = query.lower().strip(" ?!.")
    patterns = (
        r"^what\s+is\s+the\s+gwp\s+of\s+(.+)$",
        r"^what\s+is\s+gwp\s+of\s+(.+)$",
        r"^what\s+is\s+the\s+global\s+warming\s+potential\s+of\s+(.+)$",
        r"^what\s+is\s+global\s+warming\s+potential\s+of\s+(.+)$",
        r"^gwp\s+of\s+(.+)$",
        r"^global\s+warming\s+potential\s+of\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            return match.group(1).strip(" ?!.")
    return query


def normalized_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def find_column(headers: list[object], candidates: list[str]) -> int | None:
    normalized_headers = [normalized_header(header) for header in headers]
    normalized_candidates = [normalized_header(candidate) for candidate in candidates]
    for candidate in normalized_candidates:
        if candidate in normalized_headers:
            return normalized_headers.index(candidate)
    return None
