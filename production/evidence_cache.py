"""Runtime evidence cache for retrieved Class-3 strict/relaxed external evidence.

The cache is deliberately empty in the repository and is populated only from
successfully resolved live evidence during a run. It contains no predefined GWP,
density, thickness, mass-per-item value, source URL, or material-to-analog mapping.

Cache keys include the target geography so a result previously accepted for one
study geography cannot silently become Class 3 in another geography.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

CACHE_VERSION = "2.0-policy-versioned-runtime-evidence"


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


class RuntimeEvidenceCache:
    def __init__(self, path: str | os.PathLike | None = None):
        default = os.environ.get("LLM_LCA_EVIDENCE_CACHE", "")
        self.path = Path(path or default) if (path or default) else None
        self._records: dict[str, dict[str, Any]] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            # Never reuse evidence accepted under an older validation policy.
            # A code update can tighten indicator/product-identity rules, so stale
            # cached rows must be re-resolved rather than silently bypassing them.
            if not isinstance(obj, dict) or obj.get("cache_version") != CACHE_VERSION:
                self._records = {}
                return
            if isinstance(obj.get("records"), dict):
                self._records = obj["records"]
        except Exception:
            # A damaged cache must never stop a calculation.
            self._records = {}

    def _save(self) -> None:
        if not self.path:
            return
        payload = {
            "cache_version": CACHE_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "records": self._records,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        def _default(obj):
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    pass
            if isinstance(obj, Path):
                return str(obj)
            return str(obj)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def key(*, category: str, material: str, target_geography: str, property_kind: str = "") -> str:
        return "|".join([
            _norm(category), _norm(target_geography), _norm(material), _norm(property_kind)
        ])

    def get(self, *, category: str, material: str, target_geography: str, property_kind: str = "") -> dict[str, Any] | None:
        k = self.key(category=category, material=material, target_geography=target_geography, property_kind=property_kind)
        rec = self._records.get(k)
        return dict(rec.get("resolved", {})) if isinstance(rec, dict) and isinstance(rec.get("resolved"), dict) else None

    def put(self, *, category: str, material: str, target_geography: str, resolved: dict[str, Any], property_kind: str = "") -> None:
        k = self.key(category=category, material=material, target_geography=target_geography, property_kind=property_kind)
        self._records[k] = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "target_geography": target_geography,
            "material": material,
            "property_kind": property_kind,
            "resolved": dict(resolved),
        }
        self._save()

    def count(self) -> int:
        return len(self._records)
