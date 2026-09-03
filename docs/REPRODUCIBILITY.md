# Reproducibility

## Frozen study snapshot

The manuscript results are tied to the archived case-study inputs, the hash-locked ELCD retrieval catalog, the selected Qwen model revision, the frozen matching prompt/settings, and the archived output files in `case_studies/results/`.

The repository cleanup performed for GitHub does not recalculate the study outputs. It only normalizes repository/document/result filenames and removes development release clutter.

## Qwen matching protocol

The production matcher preserves the selected benchmark configuration for:

- model ID and revision;
- system prompt identity;
- character n-gram TF-IDF retrieval formulation;
- original BOM description as the retrieval query;
- Top-5 candidate pool;
- deterministic candidate presentation;
- Top-3 reported ranking;
- maximum new-token limit;
- fixed seed;
- greedy decoding with `do_sample=False`;
- 4-bit NF4 quantization; and
- structured-output parser/validator behavior.

These parameters are retained in code/lock metadata because they are needed to audit the frozen experiment. Component identifiers that look like version strings are internal provenance markers, not multiple active repository editions.

## ELCD snapshot integrity

`ELCD_Check/ELCD_Process_Catalog.xlsx` is the frozen retrieval catalog included with the study repository. `ELCD_Check/ELCD_Catalog_Lock.json` records its semantic SHA-256 and the matching-protocol settings that must agree with production.

The archived benchmark catalog is retained separately under `ELCD_Check/benchmark_reference/` for comparison with the model-selection experiment.

For the manuscript snapshot, do not overwrite the frozen catalog. A deliberately regenerated catalog represents a new environmental-data snapshot and should receive a new lock.

## Raw versus production-use matching fields

Raw Qwen fields are not silently rewritten by density lookup, unit conversion, external evidence retrieval, fallback estimation or the downstream safety gate. A separate `production_*` decision controls whether the selected ELCD process is allowed into the numerical calculation.

This preserves the distinction between:

- what the model selected; and
- what the deterministic production workflow allowed to contribute numerically.

## External evidence

External web evidence is time-dependent. Every accepted factor/property is therefore accompanied by run-level audit information such as source URL, query, extracted evidence, identity/boundary/unit checks and verification tier where applicable.

The repository does not ship a material-to-external-factor answer table.

## Monte Carlo

Monte Carlo sampling is reproducible because the random seed and convergence schedule are fixed. The raw draws archived with each case study expose the simulated row and building totals used in uncertainty analysis.

## Offline verification

From the repository root:

```bash
python scripts/self_test.py
python scripts/check_case_study_boms.py
python scripts/test_model_led_elcd_matching.py
python scripts/test_guardrails.py
python scripts/test_bounded_production.py
python scripts/test_four_class_reporting.py
python scripts/test_case_study_id_alignment.py
python scripts/test_dynamic_class4.py
python scripts/check_matching_protocol_identity.py
python scripts/check_qwen_benchmark_lock.py
```

`check_catalog_alignment.py` additionally validates the UUID universe when a quantitative openLCA factor export exists in `runtime/`.

## Environment-dependent checks

Live Qwen inference requires a CUDA-enabled runtime. The recorded benchmark used a Tesla T4-class Colab environment. Exact GPU/library reproduction provides the strongest repeatability claim, while the model revision, prompt, decoding settings and candidate presentation remain the primary protocol controls.

Rebuilding the openLCA quantitative snapshot requires:

- openLCA Desktop;
- the intended ELCD database active;
- openLCA IPC; and
- the local dependencies in `requirements-local.txt`.

## Repository integrity

`CHECKSUMS.sha256` contains SHA-256 checksums for the cleaned GitHub package. It is separate from the semantic ELCD hash stored in the catalog lock.
