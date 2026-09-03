from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from external_ef_resolver import ExternalEFResolver
from property_resolver import WebPropertyResolver, conversion_requirements
from technical_equivalence import TECHNICAL_EQUIVALENCE_SYSTEM_PROMPT


def main():
    # Conversion remains demand-driven.
    assert conversion_requirements("kg", "kg") == []
    assert conversion_requirements("t", "kg") == []
    assert conversion_requirements("m3", "m3") == []
    assert conversion_requirements("m3", "kg") == ["density_kg_m3"]
    assert conversion_requirements("m2", "kg") == ["thickness_mm", "density_kg_m3"]
    assert conversion_requirements("item", "kg") == ["mass_per_item_kg"]

    # Adaptive ceilings are algorithmic search limits, not material data.
    ef = object.__new__(ExternalEFResolver)
    prop = object.__new__(WebPropertyResolver)
    # The prompt must explicitly prohibit numerical material/property values.
    low = TECHNICAL_EQUIVALENCE_SYSTEM_PROMPT.lower()
    assert "do not provide any environmental factor" in low
    assert "density" in low and "numerical property" in low

    # Production modules must not ship value registries.
    forbidden_names = {
        "verified_source_registry.json", "verified_property_registry.json",
        "material_gwp_values.json", "material_density_values.json",
    }
    present = {p.name for p in (ROOT / "production").iterdir() if p.is_file()}
    assert not (forbidden_names & present)

    # No production module may declare the historical hard-coded data-table names.
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "production").glob("*.py"))
    for token in ("MATERIAL_GWP_TABLE", "DENSITY_TABLE", "HARDCODED_EF", "PREDEFINED_DENSITY"):
        assert token not in text

    print("adaptive/no-hardcoded-data tests: PASS")

if __name__ == "__main__":
    main()
