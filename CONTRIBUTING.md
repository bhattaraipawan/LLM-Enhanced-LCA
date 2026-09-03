# Contributing

Contributions are welcome when they preserve the project's core reproducibility rules.

- Do not add material-specific GWP lookup tables, hardcoded densities, target commercial-LCA values, or case-study correction factors to production logic.
- Keep LLM process selection constrained to supplied candidate processes.
- Keep database/source-backed results separate from terminal unverified model-only estimates.
- Add or update an offline regression test for every change to indicator, unit, product-identity, evidence-class, or cache-validation logic.
- Run `python -m compileall -q main.py production colab scripts` before opening a pull request.

Large generated runtime payloads, result ZIPs, model files, and evidence caches should not be committed.
