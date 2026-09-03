"""Qwen-only material normalization and ELCD/openLCA process matching.

The raw ELCD matching stage is benchmark-locked and model-led. A deterministic
TF-IDF retriever uses the exact benchmark algorithm on the final frozen ELCD snapshot,
and Qwen alone returns the benchmarked Direct, Proxy, or Review Required label.
The raw Qwen fields are never rewritten by calculation, density, or fallback logic.

Python applies the exact benchmark v2.3.0 schema/candidate validation. The ELCD data may be freshly exported once and hash-frozen; all matching-protocol settings remain identical. Production
then has a separate post-Qwen product-family veto that may reject an obviously
incompatible selected process for numerical use; this affects only production_*
fields and never changes the raw benchmark decision. The matching class performs
no LCIA. External evidence and fallback resolution occur afterward.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from material_taxonomy import MATERIAL_TAXONOMY_VERSION, process_compatibility
from catalog_lock import verify_lock

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PRODUCTION_PROMPT_VERSION = "4.0-exact-benchmark-protocol"
BENCHMARK_SCRIPT_VERSION = "2.3.0"
BENCHMARK_SEED = 42
BENCHMARK_CANDIDATE_POOL_SIZE = 5
BENCHMARK_TOP_K = 3
BENCHMARK_MAX_NEW_TOKENS = 256
HISTORICAL_BENCHMARK_CATALOG_CONTENT_SHA256 = "ffd076b115f91042921f82130e426da9d29d684b23ff102ef3e185b34c83d5eb"
BENCHMARK_SYSTEM_PROMPT_SHA256 = "e0ae559d973576c0fb625723f4fec6d75575c791d24f34bea7a073c51790ac70"
# Qwen's raw benchmark decision is preserved unchanged. This downstream gate may
# veto a clearly incompatible product family, but it never rewrites the raw Qwen
# Direct/Proxy/Review Required fields or chooses a replacement process.
SAFETY_GATE_VERSION = "post-qwen-family-veto-2.0"
REFERENCE_MAP_VERSION = "1.0"

SYSTEM_PROMPT = """You are an LCA material-normalization and ELCD process-matching evaluator for A1-A3 screening.

Your task is limited to material interpretation and process matching. Do NOT calculate embodied carbon and do NOT generate emission factors, GWP values, EPDs, citations, or process UUIDs. Use only the supplied candidate UUIDs.

Study definitions:
- direct: a supplied ELCD process sufficiently represents the BOM material/product.
- proxy: a supplied ELCD process is the best technically defensible substitute when an exact/direct representation is unavailable.
- review_required: none of the supplied ELCD candidates is defensible enough to select. Use this only for an unmatched material.

Rules:
1. Normalize the BOM description to a concise engineering material name.
2. Rank at most the requested number of candidate UUIDs, best first.
3. For direct or proxy, ranked_process_uuids must contain at least one supplied UUID; the first UUID is the final selected process.
4. For review_required, ranked_process_uuids must be an empty list.
5. Do not return process names, rationales, confidence scores, environmental data, or extra keys.
6. Return JSON only, with no Markdown and no text outside the JSON object.
7. Keep the response compact. Do not repeat the BOM description or candidate process names.

Required JSON schema:
{
  "normalized_material": "string",
  "ranked_process_uuids": ["uuid1", "uuid2"],
  "match_type": "direct or proxy or review_required"
}
"""


@dataclass
class MatchResult:
    item_id: str
    original_material: str
    quantity: float | None
    unit: str | None
    density_kg_m3: float | None
    thickness_mm: float | None
    mass_per_item_kg: float | None
    conversion_factor_to_ref_unit: float | None
    notes: str | None

    # Raw Qwen output (preserved exactly for audit)
    normalized_material: str | None
    match_type: str
    selected_process_uuid: str | None
    selected_process_name: str | None
    selected_process_location: str | None
    selected_process_ref_unit: str | None

    # Separate production-use fields; downstream veto never rewrites raw Qwen fields
    safety_status: str
    safety_reason: str | None
    production_approved: bool
    production_match_type: str
    production_selected_process_uuid: str | None
    production_selected_process_name: str | None
    production_selected_process_location: str | None
    production_selected_process_ref_unit: str | None

    candidate_pool_uuids: str
    candidate_pool_names: str
    candidate_pool_locations: str
    candidate_pool_scores: str
    presented_candidate_uuids: str
    parse_status: str
    structured_output_valid: bool
    usable_response: bool
    raw_model_output: str
    generation_attempts: int
    structured_output_recovered: bool
    first_parse_status: str | None
    catalog_content_sha256: str
    reference_map_sha256: str | None
    model_id: str
    model_revision: str
    production_prompt_version: str
    benchmark_script_version: str
    benchmark_seed: int
    benchmark_system_prompt_sha256: str
    safety_gate_version: str
    material_taxonomy_version: str


def _clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _norm_text(v: Any) -> str:
    s = _clean(v).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _has_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(term in text for term in terms)


def _retrieval_normalize(value: Any) -> str:
    """Match the frozen four-model benchmark TF-IDF text normalization exactly."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.%+\-/ ]", "", text)
    return text.strip()


def _search_text(row: dict[str, Any]) -> str:
    raw = (
        f"{_clean(row.get('process_name'))} | "
        f"{_clean(row.get('category'))} | "
        f"{_clean(row.get('location'))}"
    )
    return _retrieval_normalize(raw)


def _safe_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return value


def _canonical_uuid(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def _canonical_match_type(value: Any) -> str:
    text = _retrieval_normalize(value).replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"direct", "direct match", "exact", "exact match"}:
        return "direct"
    if text in {"proxy", "proxy match", "documented proxy"}:
        return "proxy"
    if text in {
        "review required", "reviewrequired", "review", "unresolved",
        "no match", "no defensible match",
    }:
        return "review_required"
    return ""


def load_catalog(path: str | Path) -> list[dict[str, Any]]:
    """Load the final frozen ELCD catalog from XLSX (preferred) or JSON.

    The XLSX uses the same ``Processes`` sheet and descriptor fields as the
    four-model benchmark exporter. The final production catalog may be freshly
    regenerated once, then its semantic hash is locked before Qwen inference.
    """
    path = Path(path)
    benchmark_cols = [
        "process_uuid", "process_name", "category", "location",
        "library", "process_type",
    ]
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, sheet_name="Processes")
        required = {"process_uuid", "process_name"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
        for col in benchmark_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str).str.strip()
        df = df[df["process_uuid"].ne("") & df["process_name"].ne("")].copy()
        if df["process_uuid"].duplicated().any():
            dupes = df.loc[df["process_uuid"].duplicated(), "process_uuid"].head(5).tolist()
            raise ValueError(f"Catalog contains duplicate process UUIDs, e.g. {dupes}")
        rows = df[benchmark_cols].to_dict(orient="records")
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Catalog JSON is empty or invalid.")
        for r in rows:
            if not {"process_uuid", "process_name"}.issubset(r):
                raise ValueError("Catalog JSON is missing process_uuid/process_name.")
            for col in benchmark_cols:
                r[col] = "" if r.get(col) is None else str(r.get(col, "")).strip()
    return rows


def content_hash(rows: list[dict[str, Any]]) -> str:
    """Exact semantic catalog hash used by benchmark script v2.3.0."""
    cols = [
        "process_uuid", "process_name", "category", "location",
        "library", "process_type",
    ]
    records = []
    for r in sorted(rows, key=lambda x: _canonical_uuid(x.get("process_uuid"))):
        records.append({col: _safe_value(r.get(col, "")) for col in cols})
    payload = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Retriever:
    def __init__(self, rows: list[dict[str, Any]], pool_size: int = 5):
        self.rows = rows
        self.pool_size = pool_size
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(
            [_search_text(r) for r in rows]
        )

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        qv = self.vectorizer.transform([_retrieval_normalize(query)])
        # TF-IDF vectors are L2-normalized, so the sparse dot product is cosine
        # similarity. Tie-breaking matches the frozen benchmark exactly.
        scores = (self.matrix @ qv.T).toarray().ravel()
        order = sorted(
            range(len(self.rows)),
            key=lambda i: (
                -float(scores[i]),
                str(self.rows[i].get("process_name", "")).lower(),
                str(self.rows[i].get("process_uuid", "")).strip().lower(),
            ),
        )[: self.pool_size]
        out = []
        for i in order:
            r = dict(self.rows[int(i)])
            r["retrieval_score"] = float(scores[int(i)])
            out.append(r)
        return out




FORMAL_CASE_STUDY_PREFIXES = (
    ("attic", "A"),
    ("stonecrete", "S"),
    ("bamboo", "B"),
)


def infer_formal_case_study_prefix(source_name: str | None) -> str | None:
    """Infer the frozen benchmark case-study ID prefix from a formal BOM filename.

    This does not affect retrieval or ELCD selection. It only preserves the same
    deterministic candidate-presentation seed used in the frozen benchmark
    (S01..., B01..., A01...) when legacy case-study BOMs contain numeric IDs.
    """
    text = str(source_name or "").strip().lower()
    for token, prefix in FORMAL_CASE_STUDY_PREFIXES:
        if token in text:
            return prefix
    return None


def _numeric_id(value: Any) -> int | None:
    """Return a positive integer for integer-like spreadsheet IDs."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        number = float(text)
    except Exception:
        return None
    if not np.isfinite(number) or number <= 0 or not float(number).is_integer():
        return None
    return int(number)


def align_formal_case_study_ids(
    df: pd.DataFrame,
    source_name: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align legacy numeric IDs to the frozen benchmark IDs for formal cases.

    Examples: Bamboo ``2`` -> ``B02``; Stonecrete ``7`` -> ``S07``; Attic
    ``1`` -> ``A01``. Existing formal IDs are normalized/preserved. Custom
    nonnumeric IDs are never overwritten. Blank IDs use the row position only
    for a recognized formal case study.

    The function intentionally changes *only* the identifier used as the
    deterministic candidate-presentation seed. Material text, retrieval scores,
    candidate pools, Qwen prompt definitions, and Direct/Proxy decisions are not
    altered.
    """
    out = df.copy()
    prefix = infer_formal_case_study_prefix(source_name)
    info: dict[str, Any] = {
        "source_name": str(source_name or ""),
        "prefix": prefix,
        "changed": 0,
        "aligned": 0,
        "custom_preserved": 0,
    }
    if prefix is None:
        return out, info

    if "ID" not in out.columns:
        out["ID"] = None

    formal_re = re.compile(rf"^{re.escape(prefix)}\s*0*(\d+)$", flags=re.I)
    new_ids: list[str] = []
    for row_pos, raw in enumerate(out["ID"].tolist(), start=1):
        text = _clean(raw)
        formal_match = formal_re.match(text) if text else None
        if formal_match:
            n = int(formal_match.group(1))
            new = f"{prefix}{n:02d}"
            info["aligned"] += 1
        else:
            n = _numeric_id(raw)
            if n is not None:
                new = f"{prefix}{n:02d}"
                info["aligned"] += 1
            elif not text:
                new = f"{prefix}{row_pos:02d}"
                info["aligned"] += 1
            else:
                new = text
                info["custom_preserved"] += 1
        if new != text:
            info["changed"] += 1
        new_ids.append(new)

    if len(set(new_ids)) != len(new_ids):
        duplicates = sorted({x for x in new_ids if new_ids.count(x) > 1})
        raise ValueError(
            "Duplicate BOM IDs after formal case-study alignment: "
            + ", ".join(duplicates)
        )
    out["ID"] = new_ids
    return out, info


def deterministic_present(
    candidates: list[dict[str, Any]],
    item_id: str,
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda r: hashlib.sha256(
            (
                "candidate-presentation-v1|"
                f"{str(item_id).strip()}|"
                f"{str(r.get('process_uuid', '')).strip().lower()}"
            ).encode("utf-8")
        ).hexdigest(),
    )


def build_benchmark_user_prompt(
    item_id: str,
    material: str,
    quantity: Any,
    unit: Any,
    presented_candidates: list[dict[str, Any]],
) -> str:
    """Exact user-prompt construction used by benchmark script v2.3.0."""
    def payload_value(value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return value

    material_payload = {
        "sample_id": str(item_id),
        "material_description": str(material),
        "quantity": payload_value(quantity),
        "unit": payload_value(unit),
    }
    compact_candidates = [
        {
            "process_uuid": r["process_uuid"],
            "process_name": r["process_name"],
            "category": r.get("category") or "",
            "location": r.get("location") or "",
            "process_type": r.get("process_type") or "",
        }
        for r in presented_candidates
    ]
    payload = {
        "requested_top_k": BENCHMARK_TOP_K,
        "material": material_payload,
        "candidate_processes": compact_candidates,
    }
    return (
        "Evaluate this material using only the supplied candidate UUIDs. "
        "The candidate list is an unordered presentation; do not infer quality from position. "
        "Return exactly the required three-field JSON object.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def extract_json(text: str) -> tuple[dict[str, Any] | None, str]:
    """Exact JSON extraction behavior from benchmark script v2.3.0."""
    if not text:
        return None, "empty_response"

    cleaned = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S | re.I)
    candidates = fenced + [cleaned]

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last > first:
        candidates.append(cleaned[first : last + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, "ok"
        except json.JSONDecodeError:
            continue
    return None, "json_parse_error"


def validate_prediction(
    parsed: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    parse_status: str,
    top_k: int = BENCHMARK_TOP_K,
) -> dict[str, Any]:
    """Exact benchmark v2.3.0 field validation/recovery semantics.

    Raw Qwen fields can therefore be compared directly with the four-model
    workbook.  Production approval is decided separately after this function.
    """
    valid_by_uuid = {
        _canonical_uuid(c["process_uuid"]): c
        for c in candidates if c.get("process_uuid")
    }
    empty = {
        "parse_status": parse_status,
        "structured_output_valid": False,
        "usable_response": False,
        "normalized_material": "",
        "normalization_field_valid": False,
        "match_type": "",
        "match_type_field_valid": False,
        "selected_process_uuid": "",
        "selected_process_name": "",
        "selection_field_valid": False,
        "ranked_process_uuids": [],
        "ranked_process_names": [],
        "ranking_field_valid": False,
        "field_recovery_used": False,
    }
    if parsed is None:
        return empty

    normalized_material = str(parsed.get("normalized_material", "")).strip()
    normalization_valid = bool(normalized_material)
    match_type = _canonical_match_type(parsed.get("match_type", ""))
    match_type_valid = match_type in {"direct", "proxy", "review_required"}

    recovery_used = False
    raw_ranked = parsed.get("ranked_process_uuids", None)
    ranked_source_valid = isinstance(raw_ranked, list)

    if not isinstance(raw_ranked, list):
        old_ranked = parsed.get("ranked_candidates", None)
        if isinstance(old_ranked, list):
            raw_ranked = [
                item.get("process_uuid", "") if isinstance(item, dict) else item
                for item in old_ranked
            ]
            recovery_used = True
        else:
            selected_old = _canonical_uuid(parsed.get("selected_process_uuid", ""))
            raw_ranked = [selected_old] if selected_old else []
            if selected_old:
                recovery_used = True

    ranked_uuids: list[str] = []
    invalid_rank_item = False
    for item in raw_ranked:
        uid = _canonical_uuid(item)
        if uid and uid in valid_by_uuid:
            if uid not in ranked_uuids:
                ranked_uuids.append(uid)
        elif uid:
            invalid_rank_item = True
        if len(ranked_uuids) >= top_k:
            break

    ranking_valid = isinstance(raw_ranked, list) and not invalid_rank_item
    if match_type == "review_required":
        ranking_valid = ranking_valid and len(ranked_uuids) == 0
        selected_uuid = ""
        selection_valid = True
    elif match_type in {"direct", "proxy"}:
        selection_valid = len(ranked_uuids) >= 1
        selected_uuid = ranked_uuids[0] if selection_valid else ""
        ranking_valid = ranking_valid and len(ranked_uuids) >= 1
    else:
        selected_uuid = ranked_uuids[0] if ranked_uuids else ""
        selection_valid = False

    selected_name = valid_by_uuid[selected_uuid]["process_name"] if selected_uuid in valid_by_uuid else ""
    ranked_names = [valid_by_uuid[u]["process_name"] for u in ranked_uuids if u in valid_by_uuid]

    exact_keys = {"normalized_material", "ranked_process_uuids", "match_type"}
    structured_valid = (
        parse_status == "ok"
        and set(parsed.keys()) == exact_keys
        and normalization_valid
        and match_type_valid
        and ranked_source_valid
        and ranking_valid
        and not recovery_used
    )
    usable = normalization_valid and match_type_valid and selection_valid and ranking_valid

    if parse_status != "ok":
        status = parse_status
    elif structured_valid:
        status = "ok"
    elif recovery_used and usable:
        status = "field_recovered"
    elif not normalization_valid:
        status = "missing_normalized_material"
    elif not match_type_valid:
        status = "invalid_match_type"
    elif not ranking_valid:
        status = "invalid_ranking"
    else:
        status = "schema_mismatch"

    return {
        "parse_status": status,
        "structured_output_valid": structured_valid,
        "usable_response": usable,
        "normalized_material": normalized_material,
        "normalization_field_valid": normalization_valid,
        "match_type": match_type,
        "match_type_field_valid": match_type_valid,
        "selected_process_uuid": selected_uuid,
        "selected_process_name": selected_name,
        "selection_field_valid": selection_valid,
        "ranked_process_uuids": ranked_uuids,
        "ranked_process_names": ranked_names,
        "ranking_field_valid": ranking_valid,
        "field_recovery_used": recovery_used,
    }


def parse_output(text: str, supplied_uuids: set[str]) -> tuple[dict[str, Any] | None, str]:
    """Compatibility wrapper used by older unit tests.

    New production matching uses :func:`extract_json` + :func:`validate_prediction`
    so the raw fields match the benchmark's validation semantics.
    """
    parsed, parse_status = extract_json(text)
    candidates = [{"process_uuid": u, "process_name": u} for u in supplied_uuids]
    pred = validate_prediction(parsed, candidates, parse_status, BENCHMARK_TOP_K)
    if not pred["usable_response"]:
        return None, pred["parse_status"]
    return {
        "normalized_material": pred["normalized_material"],
        "ranked_process_uuids": pred["ranked_process_uuids"],
        "match_type": pred["match_type"],
    }, pred["parse_status"]


def production_safety_gate(
    original_material: str,
    normalized_material: str | None,
    match_type: str,
    selected_process_name: str | None,
) -> tuple[str, str | None]:
    """Post-Qwen family veto; never changes the raw benchmark decision.

    The four-model benchmark ends after Qwen's Direct/Proxy/Review Required
    output.  Production may subsequently reject a clearly incompatible process
    (e.g. CGI -> corrugated paper board), but that rejection is recorded only in
    ``production_*`` fields and routes the row to the external/fallback resolver.
    """
    if match_type == "review_required" or not selected_process_name:
        return "NOT_APPLICABLE", None
    material_for_check = original_material or normalized_material or ""
    compatible, reason = process_compatibility(material_for_check, selected_process_name)
    if not compatible:
        return "VETO", reason
    return "PASS", "Post-Qwen product-family compatibility check passed."


class QwenMatcher:
    def __init__(
        self,
        catalog_path: str | Path,
        pool_size: int = BENCHMARK_CANDIDATE_POOL_SIZE,
        max_new_tokens: int = BENCHMARK_MAX_NEW_TOKENS,
        max_input_tokens: int = 3072,
        reference_map_path: str | Path | None = None,
    ):
        if int(pool_size) != BENCHMARK_CANDIDATE_POOL_SIZE:
            raise ValueError(
                f"Benchmark-locked production requires candidate pool size "
                f"{BENCHMARK_CANDIDATE_POOL_SIZE}; got {pool_size}."
            )
        if int(max_new_tokens) != BENCHMARK_MAX_NEW_TOKENS:
            raise ValueError(
                f"Benchmark-locked production requires max_new_tokens "
                f"{BENCHMARK_MAX_NEW_TOKENS}; got {max_new_tokens}."
            )
        actual_prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        if actual_prompt_hash != BENCHMARK_SYSTEM_PROMPT_SHA256:
            raise RuntimeError(
                "Qwen matching prompt drift detected. The production prompt no longer "
                "matches benchmark script v2.3.0."
            )
        self.catalog = load_catalog(catalog_path)
        # The ELCD data snapshot may be freshly regenerated for the final production
        # run, but it is frozen immediately afterward.  All model/prompt/retrieval
        # settings remain identical to the four-model benchmark.
        self.catalog_hash = content_hash(self.catalog)
        catalog_path_obj = Path(catalog_path)
        lock_path = catalog_path_obj.with_name("ELCD_Catalog_Lock.json")
        protocol = {
            "benchmark_script_version": BENCHMARK_SCRIPT_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "system_prompt_sha256": BENCHMARK_SYSTEM_PROMPT_SHA256,
            "seed": BENCHMARK_SEED,
            "candidate_pool_size": BENCHMARK_CANDIDATE_POOL_SIZE,
            "reported_top_k": BENCHMARK_TOP_K,
            "max_new_tokens": BENCHMARK_MAX_NEW_TOKENS,
            "temperature": 0,
            "decoding": "greedy",
            "do_sample": False,
            "quantization": "4-bit NF4",
            "retrieval_method": "character n-gram TF-IDF",
            "retrieval_analyzer": "char_wb",
            "retrieval_ngram_range": "3-5",
            "retrieval_query_source": "original BOM description only",
            "candidate_presentation": "deterministic SHA-256 shuffle by sample_id + process_uuid",
            "retrieval_rank_visible_to_llm": False,
            "retrieval_score_visible_to_llm": False,
        }
        self.catalog_lock = verify_lock(
            catalog_path_obj, lock_path, expected_matching_protocol=protocol
        )
        if str(self.catalog_lock.get("catalog_content_sha256")) != self.catalog_hash:
            raise RuntimeError("Frozen ELCD catalog lock verification failed.")
        self.reference_map_sha256 = None
        if reference_map_path is not None and Path(reference_map_path).exists():
            raw_bytes = Path(reference_map_path).read_bytes()
            self.reference_map_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            ref_map = json.loads(raw_bytes.decode("utf-8"))
            for r in self.catalog:
                meta = ref_map.get(r["process_uuid"], {})
                if meta.get("ref_unit"):
                    r["ref_unit"] = meta.get("ref_unit")
                if meta.get("qref_flow_name"):
                    r["qref_flow_name"] = meta.get("qref_flow_name")
                if meta.get("qref_amount") is not None:
                    r["qref_amount"] = meta.get("qref_amount")
        self.by_uuid = {
            r["process_uuid"]: r for r in self.catalog
        }
        self.retriever = Retriever(
            self.catalog,
            pool_size=pool_size,
        )
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.tokenizer = None
        self.model = None

    @staticmethod
    def _set_benchmark_reproducibility() -> None:
        """Mirror benchmark script v2.3.0 seed/determinism settings exactly."""
        import torch
        random.seed(BENCHMARK_SEED)
        np.random.seed(BENCHMARK_SEED)
        torch.manual_seed(BENCHMARK_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(BENCHMARK_SEED)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def load_model(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is not available. In Colab choose "
                "Runtime > Change runtime type > T4 GPU."
            )

        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self._set_benchmark_reproducibility()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_fast=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            quantization_config=qconf,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        loaded_revision = str(getattr(self.model.config, "_commit_hash", "") or "")
        if loaded_revision and loaded_revision != MODEL_REVISION:
            raise RuntimeError(
                f"Qwen model revision drift: loaded {loaded_revision}, expected {MODEL_REVISION}."
            )
        return self

    def generate_with_system(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        max_new_tokens_override: int | None = None,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                ),
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

        try:
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        int(max_new_tokens_override)
                        if max_new_tokens_override is not None
                        else self.max_new_tokens
                    ),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = out[0][inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()
            return decoded
        finally:
            # Release per-generation tensors promptly. This does not unload the model.
            try:
                del inputs
                if "out" in locals():
                    del out
                if "generated" in locals():
                    del generated
            except Exception:
                pass
            torch.cuda.empty_cache()

    def _generate_matching_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens_override: int | None = None,
    ) -> str:
        """Generate ELCD matching output using the frozen benchmark chat format.

        The four-model benchmark used one user-role message containing the system
        instruction followed by the serialized benchmark payload. Production uses
        the same format here so the selected Qwen sees the same matching protocol.
        Other downstream resolver prompts continue to use ``generate_with_system``.
        """
        import torch

        self._set_benchmark_reproducibility()
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        try:
            device = self.model.get_input_embeddings().weight.device
        except Exception:
            device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        try:
            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=(
                        int(max_new_tokens_override)
                        if max_new_tokens_override is not None
                        else self.max_new_tokens
                    ),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated = out[0, input_ids.shape[-1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        finally:
            try:
                del input_ids
                del attention_mask
                if "out" in locals():
                    del out
                if "generated" in locals():
                    del generated
            except Exception:
                pass
            torch.cuda.empty_cache()

    def _generate(self, user_prompt: str) -> str:
        return self._generate_matching_prompt(SYSTEM_PROMPT, user_prompt)

    def match_item(self, item: dict[str, Any]) -> MatchResult:
        item_id = _clean(
            item.get("ID")
            or item.get("item_id")
            or item.get("id")
            or "ITEM"
        )
        material = _clean(
            item.get("Material") or item.get("material")
        )
        if not material:
            raise ValueError("Material is required.")

        candidates = self.retriever.retrieve(material)
        presented = deterministic_present(
            candidates,
            item_id,
        )

        user_prompt = build_benchmark_user_prompt(
            item_id, material, item.get("Quantity"), item.get("Unit"), presented
        )

        # Exact benchmark inference protocol: one generation, no production-only
        # repair prompt.  Raw fields are validated with the same v2.3.0 logic used
        # in the four-model workbook.
        raw = self._generate(user_prompt)
        parsed_obj, json_status = extract_json(raw)
        prediction = validate_prediction(
            parsed_obj, presented, json_status, BENCHMARK_TOP_K
        )

        normalized = prediction["normalized_material"] or None
        match_type = prediction["match_type"] or "review_required"
        selected_uuid = prediction["selected_process_uuid"] or None
        selected = self.by_uuid.get(selected_uuid) if selected_uuid else None
        selected_name = selected.get("process_name") if selected else None
        selected_location = selected.get("location") if selected else None
        selected_ref_unit = selected.get("ref_unit") if selected else None

        structured_valid = bool(prediction["structured_output_valid"])
        usable_response = bool(prediction["usable_response"])
        status = str(prediction["parse_status"])

        safety_status, safety_reason = production_safety_gate(
            original_material=material,
            normalized_material=normalized,
            match_type=match_type,
            selected_process_name=selected_name,
        )

        # The raw Qwen fields above are never rewritten. Production acceptance is
        # a separate downstream decision. For manuscript-grade traceability we
        # require the exact benchmark schema to be valid and the family gate to
        # pass before an ELCD factor is used.
        production_approved = (
            structured_valid
            and usable_response
            and match_type in {"direct", "proxy"}
            and bool(selected_uuid)
            and safety_status == "PASS"
        )

        if production_approved:
            production_match_type = match_type
            production_uuid = selected_uuid
            production_name = selected_name
            production_location = selected_location
            production_ref_unit = selected_ref_unit
        else:
            production_match_type = "review_required"
            production_uuid = None
            production_name = None
            production_location = None
            production_ref_unit = None

        generation_attempts = 1
        structured_output_recovered = False
        first_parse_status = status

        def num(name: str):
            v = item.get(name)
            if (
                v is None
                or (
                    isinstance(v, float)
                    and np.isnan(v)
                )
                or _clean(v) == ""
            ):
                return None
            try:
                return float(v)
            except Exception:
                return None

        return MatchResult(
            item_id=item_id,
            original_material=material,
            quantity=num("Quantity"),
            unit=_clean(item.get("Unit")) or None,
            density_kg_m3=num("Density_kg_m3"),
            thickness_mm=num("Thickness_mm"),
            mass_per_item_kg=num("Mass_per_item_kg"),
            conversion_factor_to_ref_unit=num(
                "Conversion_factor_to_ref_unit"
            ),
            notes=_clean(item.get("Notes")) or None,

            normalized_material=normalized,
            match_type=match_type,
            selected_process_uuid=selected_uuid,
            selected_process_name=selected_name,
            selected_process_location=selected_location,
            selected_process_ref_unit=selected_ref_unit,

            safety_status=safety_status,
            safety_reason=safety_reason,
            production_approved=production_approved,
            production_match_type=production_match_type,
            production_selected_process_uuid=production_uuid,
            production_selected_process_name=production_name,
            production_selected_process_location=production_location,
            production_selected_process_ref_unit=production_ref_unit,

            candidate_pool_uuids=json.dumps(
                [r["process_uuid"] for r in candidates]
            ),
            candidate_pool_names=json.dumps(
                [r["process_name"] for r in candidates],
                ensure_ascii=False,
            ),
            candidate_pool_locations=json.dumps(
                [r.get("location") for r in candidates],
                ensure_ascii=False,
            ),
            candidate_pool_scores=json.dumps(
                [
                    round(r["retrieval_score"], 8)
                    for r in candidates
                ]
            ),
            presented_candidate_uuids=json.dumps(
                [r["process_uuid"] for r in presented]
            ),
            parse_status=status,
            structured_output_valid=structured_valid,
            usable_response=usable_response,
            raw_model_output=raw,
            generation_attempts=generation_attempts,
            structured_output_recovered=structured_output_recovered,
            first_parse_status=first_parse_status,
            catalog_content_sha256=self.catalog_hash,
            reference_map_sha256=self.reference_map_sha256,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            production_prompt_version=PRODUCTION_PROMPT_VERSION,
            benchmark_script_version=BENCHMARK_SCRIPT_VERSION,
            benchmark_seed=BENCHMARK_SEED,
            benchmark_system_prompt_sha256=BENCHMARK_SYSTEM_PROMPT_SHA256,
            safety_gate_version=SAFETY_GATE_VERSION,
            material_taxonomy_version=MATERIAL_TAXONOMY_VERSION,
        )

    def match_many(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"Material", "Quantity", "Unit"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                f"BOM is missing required columns: {missing}"
            )

        rows = []
        for i, rec in enumerate(
            df.to_dict(orient="records"),
            start=1,
        ):
            if not _clean(rec.get("ID")):
                rec["ID"] = f"ITEM-{i:03d}"
            rows.append(asdict(self.match_item(rec)))

        return pd.DataFrame(rows)


def write_selection_excel(
    df: pd.DataFrame,
    path: str | Path,
    property_evidence: pd.DataFrame | None = None,
    external_ef_evidence: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            sheet_name="Selections",
            index=False,
        )

        summary = pd.DataFrame(
            [
                ["n_items", len(df)],
                [
                    "structured_output_valid",
                    int(df["structured_output_valid"].sum()),
                ],
                [
                    "structured_output_recovered_after_retry",
                    int(df.get("structured_output_recovered", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()),
                ],
                [
                    "qwen_direct_raw",
                    int((df["match_type"] == "direct").sum()),
                ],
                [
                    "qwen_proxy_raw",
                    int((df["match_type"] == "proxy").sum()),
                ],
                [
                    "qwen_review_required_raw",
                    int(
                        (
                            df["match_type"]
                            == "review_required"
                        ).sum()
                    ),
                ],
                [
                    "safety_flagged",
                    int(
                        (
                            df["safety_status"].isin({"VETO", "FLAGGED"})
                        ).sum()
                    ),
                ],
                [
                    "production_approved",
                    int(df["production_approved"].sum()),
                ],
                [
                    "production_direct",
                    int(
                        (
                            df["production_match_type"]
                            == "direct"
                        ).sum()
                    ),
                ],
                [
                    "production_proxy",
                    int(
                        (
                            df["production_match_type"]
                            == "proxy"
                        ).sum()
                    ),
                ],
                [
                    "production_review_required",
                    int(
                        (
                            df["production_match_type"]
                            == "review_required"
                        ).sum()
                    ),
                ],
                [
                    "catalog_content_sha256",
                    df["catalog_content_sha256"].iloc[0]
                    if len(df)
                    else None,
                ],
                ["model_id", MODEL_ID],
                ["model_revision", MODEL_REVISION],
                [
                    "production_prompt_version",
                    PRODUCTION_PROMPT_VERSION,
                ],
                [
                    "safety_gate_version",
                    SAFETY_GATE_VERSION,
                ],
                [
                    "material_taxonomy_version",
                    MATERIAL_TAXONOMY_VERSION,
                ],
            ],
            columns=["field", "value"],
        )

        # External-EF metadata are appended only when the resolver ran.
        if "external_ef_resolution_status" in df.columns:
            ef_rows = pd.DataFrame([
                ["external_ef_resolver_version", df["external_ef_resolver_version"].dropna().iloc[0] if df["external_ef_resolver_version"].notna().any() else None],
                ["external_ef_verified", int((df["external_ef_resolution_status"] == "RESOLVED_EXTERNAL_VERIFIED").sum())],
                ["external_ef_verified_relaxed_phase", int((df.get("external_ef_verification_tier", pd.Series(index=df.index, dtype=str)) == "RELAXED").sum())],
                ["external_ef_unresolved", int((df["external_ef_resolution_status"] == "UNRESOLVED").sum())],
                ["external_ef_not_applicable", int((df["external_ef_resolution_status"] == "NOT_APPLICABLE_ELCD_APPROVED").sum())],
            ], columns=["field", "value"])
            summary = pd.concat([summary, ef_rows], ignore_index=True)

        # Property-resolution metadata are appended only when the property resolver ran.
        if "property_resolution_status" in df.columns:
            prop_rows = pd.DataFrame([
                ["property_resolver_version", df["property_resolver_version"].dropna().iloc[0] if df["property_resolver_version"].notna().any() else None],
                ["property_traceable_web", int((df["property_resolution_status"] == "RESOLVED_TRACEABLE_WEB").sum())],
                ["property_traceable_web_relaxed", int((df["property_resolution_status"] == "RESOLVED_TRACEABLE_WEB_RELAXED").sum())],
                ["property_unresolved", int((df["property_resolution_status"] == "UNRESOLVED").sum())],
                ["property_not_needed", int((df["property_resolution_status"] == "NOT_NEEDED").sum())],
            ], columns=["field", "value"])
            summary = pd.concat([summary, prop_rows], ignore_index=True)

        summary.to_excel(
            writer,
            sheet_name="Run_Metadata",
            index=False,
        )

        if property_evidence is not None and not property_evidence.empty:
            property_evidence.to_excel(
                writer,
                sheet_name="Property_Evidence",
                index=False,
            )

        if external_ef_evidence is not None and not external_ef_evidence.empty:
            external_ef_evidence.to_excel(
                writer,
                sheet_name="External_EF_Evidence",
                index=False,
            )

    return path
