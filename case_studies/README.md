# Case-study data

This directory contains the three standardized BOM inputs used in the manuscript:

- `BOM_Stonecrete_Template.xlsx`
- `BOM_Bamboo_Template.xlsx`
- `BOM_Attic_Template.xlsx`

Required BOM columns are `Material`, `Quantity`, and `Unit`; `ID` is recommended.

The formal study identifiers use `S..`, `B..`, and `A..` prefixes so candidate presentation remains aligned with the controlled benchmark protocol. These identifiers affect reproducible candidate ordering only; they do not encode ELCD selections or environmental values.

## Frozen manuscript outputs

`results/` contains the archived outputs from the study runs completed on **30 August 2026**. Repository cleanup changed only filenames and folder organization; it did not recalculate or edit the numerical workbook contents.

Each building folder contains:

- the A1–A3 GWP workbook;
- the Verified contribution figure;
- the Complete exploratory contribution figure; and
- the compressed raw Monte Carlo draws.

Nested duplicate result ZIPs from the original run packages were intentionally omitted to keep the GitHub repository compact.
