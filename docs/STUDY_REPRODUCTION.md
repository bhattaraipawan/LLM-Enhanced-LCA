# Applying the workflow to a new ELCD snapshot

The repository already contains the frozen catalog and archived outputs used for the manuscript. **Do not run the export step simply to reproduce the reported results.** Use this procedure only when intentionally applying the framework to a newly exported environmental-data snapshot.

## 1. Prepare openLCA

Activate the intended ELCD database in openLCA and start the IPC server (default port `8080`). Keep the same database active for the catalog, reference-unit and GWP-factor exports.

## 2. Install local dependencies

```bash
python -m pip install -r requirements-local.txt
```

## 3. Export and lock the data snapshot

```bash
python main.py export
```

This command:

1. exports the process catalog;
2. writes a semantic catalog lock;
3. exports the process/reference-unit map;
4. calculates the characterized GWP snapshot;
5. checks process-UUID consistency; and
6. creates `runtime/Qwen_Colab_Payload.zip` with integrity metadata.

A regenerated catalog is a **new data snapshot**. Do not describe its results as an exact reproduction of the archived manuscript case studies unless the relevant frozen hashes and inputs are identical.

## 4. Verify protocol identity

```bash
python scripts/check_matching_protocol_identity.py
python scripts/check_matching_algorithm_identity.py
python scripts/check_catalog_alignment.py
python scripts/check_qwen_benchmark_lock.py
```

These checks distinguish matching-protocol identity from environmental-data changes.

## 5. Run in Colab

Open `colab/LLM_LCA_Colab.ipynb` or execute `colab/LLM_LCA_Colab.py` in Google Colab. Use a CUDA GPU, upload `runtime/Qwen_Colab_Payload.zip`, then process the desired BOM workbooks.

For the three formal manuscript case studies, use the standardized BOM templates in `case_studies/` and the same payload for all buildings.
