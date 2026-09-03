# Production workflow

This document describes the frozen manuscript workflow. It is intended to match the implementation in `production/` and the archived case-study outputs.

## 1. Input

A BOM requires `Material`, `Quantity`, and `Unit`; `ID` is recommended. Optional project-supplied conversion fields include `Density_kg_m3`, `Thickness_mm`, `Mass_per_item_kg`, and `Conversion_factor_to_ref_unit`.

## 2. ELCD candidate retrieval

The original BOM description is converted to character n-gram TF-IDF features (`char_wb`, 3–5 grams). The five highest-ranked ELCD candidates form the bounded candidate pool supplied to Qwen. Retrieval is deterministic for a fixed catalog and input.

## 3. Qwen process-selection task

Qwen2.5-7B-Instruct performs three bounded tasks:

1. normalize the BOM material description;
2. rank the supplied candidate processes; and
3. return Direct, Proxy, or Review Required.

The model cannot select a UUID outside the candidate pool. Structured output is validated for schema integrity, candidate membership, ranking consistency, and decision consistency. Raw model fields are retained in the audit output.

A downstream production-use gate may veto an obviously incompatible product family before numerical calculation. It does not choose a replacement process and does not rewrite the raw Qwen decision.

## 4. Evidence hierarchy

### Class 1 — ELCD Direct

A directly representative ELCD process is used with its frozen characterized GWP factor.

### Class 2 — ELCD Proxy

A technically defensible ELCD proxy is used with its frozen characterized GWP factor.

### Class 3 — External Verified

When ELCD does not provide an accepted numerical route, external evidence is searched dynamically.

**Phase A — strict evidence.** Target geography and direct product identity are prioritized, with A1–A3 product-stage reporting preferred.

**Phase B — relaxed evidence.** If Phase A fails, geography and technical equivalence may broaden. The numerical value must still be explicitly supported by retrievable evidence and pass product/family, indicator, unit, product-stage/boundary, and evidence-support checks.

Both accepted phases are reported as External Verified. The retrieval phase and representativeness limitations remain auditable.

## 5. Class 4 fallback

Class 4 is used only after Classes 1–3 fail. The resolver first attempts database-anchored dynamic same-family and semantic-analog routes using the frozen ELCD/openLCA snapshot. A terminal model-only estimate is allowed only after those routes fail. Any such value is explicitly unverified and excluded from the verified subtotal.

## 6. Unit conversion and properties

Python performs unit conversion deterministically. Project/BOM conversion properties have highest priority. When a required property is absent, traceable evidence is sought first; an unverified required property moves the affected line to Class 4.

No fixed material-specific density or physical-property registry is embedded in the production code.

## 7. Calculation

Quantity conversion, multiplication, aggregation and reporting are deterministic. For line item `i`:

```text
GWP_i = quantity_in_reference_unit_i × emission_factor_i
```

The two nested building populations are:

```text
Verified subtotal = Classes 1 + 2 + 3
Complete exploratory estimate = Classes 1 + 2 + 3 + 4
```

If a row cannot produce a numerical result, it is reported as an input/model failure rather than silently assigned a value.

## 8. Biogenic reporting

For timber, plywood and bamboo, storage/sequestration credit is excluded from the reported A1–A3 screening result. The workflow seeks a positive compatible GWP reporting basis according to the implemented policy.

## 9. Uncertainty

Row uncertainty is determined from the strongest available information: source-reported uncertainty, dispersion among accepted values, or an adapted pedigree method. Uncertainty associated with required conversion properties is propagated with factor uncertainty.

Monte Carlo sampling uses independent lognormal multipliers and a fixed seed of 42. Candidate run counts are 1,000, 5,000, 10,000, 25,000 and 50,000, with a minimum retained run count of 10,000 and a 1% convergence criterion applied to the reported building populations.

## 10. Formal case-study identifiers

The formal Stonecrete, Bamboo and Attic inputs use `S..`, `B..` and `A..` identifiers. The identifier alignment supports deterministic candidate presentation and does not encode a process answer, emission factor, density, or material-specific correction.
