# LLM-Enhanced LCA

Open-source workflow for LLM-assisted **A1–A3 upfront embodied-carbon screening** from building bills of materials (BOMs).

This repository is the **frozen study implementation** used for the manuscript case studies. The scientific workflow is intentionally separated into: (1) deterministic candidate retrieval, (2) bounded LLM-assisted material interpretation and ELCD process selection, (3) downstream environmental-evidence resolution when ELCD does not provide a defensible numerical result, and (4) deterministic unit conversion, GWP calculation, aggregation, reporting, and uncertainty propagation.

## Study status

- Production model: **Qwen2.5-7B-Instruct**.
- System boundary: **A1–A3 (cradle-to-gate)**.
- ELCD catalog used by the frozen study snapshot: **608 processes**.
- Retrieval: character n-gram TF-IDF (`char_wb`, 3–5 grams), using the original BOM description.
- Candidate pool: **Top-5** ELCD processes.
- Qwen output: normalized material, ranked candidates, and **Direct / Proxy / Review Required** decision.
- Numerical calculations: deterministic Python operations only.
- Monte Carlo seed: **42**.
- Biogenic reporting policy: storage/sequestration credit is excluded for timber, plywood, and bamboo.

The model revision, prompt hash, catalog hash, component audit identifiers, and benchmark settings retained in the code and lock files are **reproducibility fingerprints**, not separate active code releases.

## Evidence hierarchy used in the manuscript

1. **Class 1 — ELCD Direct (`ELCD_DIRECT`)**  
   A directly representative ELCD process is accepted from the supplied Top-5 candidate pool.

2. **Class 2 — ELCD Proxy (`ELCD_PROXY`)**  
   A technically defensible ELCD proxy is accepted from the supplied Top-5 candidate pool.

3. **Class 3 — External Verified (`EXTERNAL_VERIFIED`)**  
   When ELCD matching does not yield a usable numerical result, traceable external evidence is searched at run time. The resolver first prioritizes target-geography/direct-product evidence and then, if necessary, broadens geography and technical equivalence while retaining the phase and evidence limitations in the audit trail.

4. **Class 4 — Unverified Fallback Estimate (`UNVERIFIED_FALLBACK_ESTIMATE`)**  
   Used only after Classes 1–3 fail. Database-anchored dynamic analog routes are attempted before any terminal model-only estimate. Class 4 remains explicitly unverified.

The publication-facing building totals are:

- **Verified A1–A3 GWP subtotal:** Classes 1–3.
- **Complete exploratory GWP estimate:** Classes 1–4.

`INPUT_OR_MODEL_FAILURE` is an execution status, not an evidence class.

## No predetermined environmental answers

The production code does **not** contain:

- material-specific GWP lookup tables;
- fixed density registries;
- preselected external sources;
- case-study correction factors;
- One Click LCA target totals; or
- material-specific rules that force a particular ELCD process.

Project/BOM conversion properties, when supplied, take precedence. Missing conversion properties are resolved dynamically using the same evidence philosophy as the environmental-factor pathway.

## Repository structure

```text
LLM-Enhanced-LCA/
├── main.py                         Local openLCA/CLI entry point
├── production/                     Production matching, evidence, calculation and uncertainty modules
├── colab/                          Standalone Google Colab workflow
├── case_studies/
│   ├── BOM_*_Template.xlsx         Frozen case-study BOM inputs
│   └── results/                    Frozen manuscript outputs
├── ELCD_Check/                     Frozen ELCD catalog, lock and expert-reference materials
├── Four_Models/                    Four-model benchmark inputs, archived outputs and model-selection analysis
├── scripts/                        Reproducibility, benchmark and offline verification utilities
├── docs/                           Workflow and manuscript-facing documentation
├── requirements*.txt
├── CITATION.cff
├── LICENSE
└── CHECKSUMS.sha256
```

## Core workflow

```text
BOM material description
        ↓
character n-gram TF-IDF retrieval
        ↓
Top-5 ELCD candidates
        ↓
Qwen normalization + candidate ranking + Direct / Proxy / Review Required
        ↓
structured-output and candidate-pool validation
        ↓
post-Qwen production-use safety gate
        ↓
ELCD Direct / ELCD Proxy
        or
External Verified evidence resolution
        or
Unverified Fallback Estimate
        ↓
deterministic unit conversion and line-item GWP
        ↓
Verified subtotal + Complete exploratory estimate
        ↓
row-level uncertainty + Monte Carlo propagation
```

Raw Qwen selection fields are retained for audit. Downstream safety/evidence logic may prevent an incompatible selected ELCD process from entering the numerical calculation, but it does not silently replace or rewrite the raw model decision.

## Frozen case-study inputs and results

The manuscript case studies are provided under `case_studies/`:

- Stonecrete House;
- Bamboo House; and
- Bamboo House with Attic.

The corresponding frozen outputs are under `case_studies/results/`. Filenames were normalized for GitHub readability; the workbook contents and numerical results were not recalculated or altered during repository cleanup.

## Reproducing the frozen study configuration

For verification of the repository snapshot:

```bash
python -m pip install -r requirements-local.txt
python scripts/validate_repository.py
```

The validation helper runs the core offline checks plus protocol/lock checks. Live Qwen inference requires a CUDA-enabled environment. Re-exporting an ELCD/openLCA snapshot requires openLCA Desktop, the intended ELCD database, and the IPC server.

To reproduce the **reported manuscript results**, use the frozen case-study inputs, frozen catalog/lock, and archived result files in this repository. Do not modify the production logic or overwrite the frozen catalog.

To apply the framework to a **new environmental-data snapshot**, follow `docs/STUDY_REPRODUCTION.md` and create a new catalog lock and Colab payload.

## Google Colab

The Colab workflow is available as both:

- `colab/LLM_LCA_Colab.ipynb`; and
- `colab/LLM_LCA_Colab.py`.

The local preparation stage builds `runtime/Qwen_Colab_Payload.zip`, which contains the frozen ELCD retrieval catalog, reference-unit map, characterized GWP snapshot, production modules, and integrity manifest. The payload is then uploaded to the Colab runner.

## Outputs

For each processed BOM, the workflow produces a workbook, two contribution figures, raw Monte Carlo draws, and a result package. Principal workbook sheets include:

- `Summary`;
- `Verified_Calculation`;
- `Complete_Exploratory_Screening`;
- `Evidence_Audit`;
- `Uncertainty_Summary`;
- `Uncertainty_By_Row`;
- `MC_Convergence`;
- `Selections_Audit`;
- `Run_Metadata`; and
- evidence-detail sheets when external factors or conversion properties are resolved.

The publication contribution figures intentionally contain no chart title, no Direct/Proxy symbol, and no evidence-class legend. Materials contributing less than 0.1% individually are combined into `Other`.

## Manuscript terminology

For wording that matches the frozen implementation, see `docs/MANUSCRIPT_REPORTING.md`.

## Reproducibility and citation

See:

- `docs/PRODUCTION_WORKFLOW.md`;
- `docs/REPRODUCIBILITY.md`;
- `docs/REPRODUCIBILITY_CHECKLIST.md`; and
- `CITATION.cff`.

The repository is licensed under the MIT License.
