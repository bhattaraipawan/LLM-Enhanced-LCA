# ELCD catalog and expert-reference materials

This directory contains the ELCD process data used for matching and the human expert-reference materials used for the controlled four-model benchmark.

## Files

```text
ELCD_Check/
├── ELCD_Process_Catalog.xlsx
├── ELCD_Catalog_Lock.json
├── expert_reference/
│   └── LLM_LCA_Expert_Reference_Set_With_ELCD.xlsx
└── benchmark_reference/
    └── ELCD_Process_Catalog_Benchmark.xlsx
```

## Frozen production catalog

`ELCD_Process_Catalog.xlsx` contains 608 process descriptors from the study ELCD snapshot. The production matcher retrieves candidates from this catalog.

`ELCD_Catalog_Lock.json` records:

- the semantic catalog SHA-256;
- process count;
- selected Qwen model/revision;
- prompt identity;
- seed and decoding settings;
- candidate-pool and reported-ranking sizes; and
- deterministic retrieval/presentation settings.

Production verification fails if the current catalog content does not match its lock.

## Benchmark reference catalog

`benchmark_reference/ELCD_Process_Catalog_Benchmark.xlsx` is retained only to audit the catalog used by the completed model-selection experiment. It is not a second production catalog.

## Expert reference

`expert_reference/LLM_LCA_Expert_Reference_Set_With_ELCD.xlsx` contains the independently reviewed/reconciled benchmark reference material. The evaluated LLMs are not used to manufacture the expert labels.

## Regenerating a catalog

For a deliberate new environmental-data snapshot, activate the intended ELCD database in openLCA and run the export procedure described in `../docs/STUDY_REPRODUCTION.md`.

Do not overwrite the frozen manuscript catalog when the goal is to reproduce the archived study results.
