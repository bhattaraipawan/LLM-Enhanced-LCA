# Reproducibility checklist

Use this checklist when auditing the manuscript implementation or creating a deliberate new environmental-data snapshot.

## Frozen manuscript snapshot

- [ ] Use the BOM templates in `case_studies/`.
- [ ] Keep `ELCD_Check/ELCD_Process_Catalog.xlsx` unchanged.
- [ ] Confirm the catalog semantic hash matches `ELCD_Check/ELCD_Catalog_Lock.json`.
- [ ] Use the pinned Qwen model/revision and matching settings in the production matcher.
- [ ] Preserve the formal `S..`, `B..` and `A..` case-study IDs.
- [ ] Do not inject commercial-LCA target values, material-specific GWP tables or fixed density answers.
- [ ] Keep raw model-selection fields distinct from downstream production-use fields.
- [ ] Report Classes 1–3 as the Verified subtotal and Classes 1–4 as the Complete exploratory estimate.
- [ ] Exclude biogenic storage/sequestration credit under the implemented policy.

## Offline checks

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

## New ELCD snapshot only

When intentionally exporting a new snapshot:

```bash
python main.py export
python scripts/check_matching_protocol_identity.py
python scripts/check_matching_algorithm_identity.py
python scripts/check_catalog_alignment.py
```

A different catalog hash is a new environmental-data snapshot, not a matching-protocol change. Keep the new outputs separate from the archived manuscript results.
