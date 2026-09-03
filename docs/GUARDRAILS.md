# Production guardrails

These guardrails apply downstream of the raw Qwen ELCD decision unless stated otherwise. Their purpose is to prevent unsupported numerical use without silently replacing the model output.

## Structured-output integrity

The matching output is checked for valid structure, candidate-pool UUID membership, ranking consistency and decision consistency. Qwen cannot select an ELCD UUID outside the supplied candidate pool.

## Product-family production-use gate

A selected ELCD process can be vetoed for numerical use when its product family is clearly incompatible with the BOM material. The gate does not choose another process and does not rewrite the raw Qwen selection.

## GWP indicator identity

For ordinary non-biogenic materials, explicit component indicators such as biogenic or land-use components are not accepted as substitutes for the required total/product-stage GWP basis. Bio-based materials follow the study policy that excludes storage/sequestration credit.

## External evidence validation

Accepted external factors must pass checks for source/evidence support, product or product-family identity, indicator, declared/reference unit and product-stage/boundary compatibility. Representativeness limitations from relaxed retrieval remain in the audit trail and uncertainty assignment.

## Terminal Class-4 identity

A model-only Class-4 estimate is reached only after source-supported and database-anchored routes fail. Material identity must remain compatible with the BOM taxonomy. Rationale text cannot override an incompatible product interpretation.

## Runtime evidence cache

The cache contains only evidence accepted during execution. A policy identifier is stored with cached evidence so that data accepted under incompatible validation rules are not silently reused.

## No target tuning

The guardrails do not contain One Click LCA totals, case-study target values, material-specific GWP tables, fixed material densities or target numerical ranges.
