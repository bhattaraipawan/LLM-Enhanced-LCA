"""Audit the frozen case-study BOMs against the downstream production material taxonomy.

This QA script does not calculate GWP and does not require network access.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))
from material_taxonomy import MATERIAL_TAXONOMY_VERSION, classify_material

CASE_STUDIES = [
    ROOT / "case_studies/BOM_Stonecrete_Template.xlsx",
    ROOT / "case_studies/BOM_Bamboo_Template.xlsx",
    ROOT / "case_studies/BOM_Attic_Template.xlsx",
]

def main():
    unknown = []
    print(f"Material taxonomy version: {MATERIAL_TAXONOMY_VERSION}")
    for path in CASE_STUDIES:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_excel(path, sheet_name="BOM_Input")
        df = df[df["Material"].notna()].copy()
        print(f"\n{path.name}: {len(df)} materials")
        for _, row in df.iterrows():
            material = str(row["Material"]).strip()
            family = classify_material(material)
            print(f"  {material:<60} -> {family}")
            if family == "UNKNOWN":
                unknown.append((path.name, material))
    if unknown:
        print("\nUNKNOWN MATERIALS:")
        for file_name, material in unknown:
            print(f" - {file_name}: {material}")
        raise SystemExit(2)
    print("\nPASS: all case-study BOM materials are covered by the downstream material taxonomy.")
    print("No material-specific emission-factor or physical-property registry is used.")

if __name__ == "__main__":
    main()
