# Manuscript reporting language

The statements below are aligned with the frozen repository and can be used as a consistency reference when revising the manuscript.

## System boundary

The assessment is an **A1–A3 cradle-to-gate upfront embodied-carbon screening** workflow.

## LLM role

Qwen2.5-7B-Instruct is used for material-description normalization, ranking of a deterministic Top-5 ELCD candidate set, and assignment of Direct / Proxy / Review Required. The model is not used for deterministic quantity conversion, multiplication, aggregation or building-level GWP calculation.

## Candidate retrieval

ELCD candidates are retrieved using character n-gram TF-IDF (`char_wb`, 3–5 grams) from the original BOM material description. Qwen is constrained to the supplied candidate pool.

## Evidence classes

Use these publication labels consistently:

- **Class 1 — ELCD Direct**
- **Class 2 — ELCD Proxy**
- **Class 3 — External Verified**
- **Class 4 — Unverified Fallback Estimate**

Do not describe Class 4 as verified or source-supported merely because a database-anchored analog route was attempted before the terminal fallback.

## Building-level outputs

Use:

- **Verified A1–A3 GWP subtotal** for Classes 1–3; and
- **Complete exploratory GWP estimate** for Classes 1–4.

If numerical coverage is incomplete, do not call the broadest subtotal a complete estimate.

## openLCA/ELCD role

openLCA is used locally to export/characterize the ELCD data snapshot. The Colab production calculation applies the frozen catalog/reference-unit/GWP snapshot in Python rather than performing a live openLCA calculation for each BOM row.

## External factors and conversion properties

When ELCD does not provide a defensible numerical route, external evidence is searched dynamically. Target-geography/direct-product evidence is attempted before broader geography/technical-equivalence evidence. Missing conversion properties are handled dynamically; the production code does not contain a fixed material-density registry.

## Biogenic policy

For timber, plywood and bamboo, storage/sequestration credit is excluded from the reported A1–A3 screening result.

## Uncertainty

Uncertainty is assigned at row level from source-reported information, accepted-value dispersion or an adapted pedigree approach, and propagated through Monte Carlo simulation with a fixed seed and convergence check. The resulting interval represents the implemented parameter/evidence uncertainty model and does not convert Class-4 values into verified data.

## Reproducibility claim

The strongest defensible claim is that the manuscript results are reproducible from the **frozen repository snapshot, frozen case-study inputs, locked ELCD catalog/data payload, pinned Qwen model/revision and recorded matching settings**, subject to the known time dependence of external web evidence when rerunning the retrieval stage.
