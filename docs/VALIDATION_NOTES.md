# Validation notes

The repository keeps the model-selection task separate from downstream numerical evidence resolution.

## Matching-stage validation

The raw Direct / Proxy / Review Required decision is model-led within the supplied Top-5 candidate pool. Deterministic matching-stage checks validate structure, UUID membership, ranking and decision consistency. They do not substitute a hand-coded process answer.

## Downstream numerical validation

Before an ELCD process contributes numerically, a separate production-use gate can veto a clearly incompatible product family. If no ELCD numerical route is accepted, the workflow proceeds to External Verified and then, if necessary, Unverified Fallback Estimate.

Class 3 remains part of the verified subtotal only when the retrieved factor/property satisfies the implemented evidence checks. Class 4 is explicitly unverified and is included only in the complete exploratory estimate.

## Offline regression coverage

The included tests check, among other things:

- malformed or contradictory structured model output;
- selection outside the supplied candidate pool;
- deterministic retrieval/presentation behavior;
- absence of predetermined GWP/property answer registries;
- product-family and indicator safeguards;
- bounded external-evidence behavior;
- separation of Classes 1–4;
- dynamic Class-4 fallback behavior;
- case-study identifier alignment;
- uncertainty propagation and reporting labels; and
- incomplete-result handling when numerical coverage fails.

Run the core offline checks listed in the main `README.md`.

Live Qwen inference, web retrieval and openLCA export are environment-dependent and are not fully exercised by the offline tests.
