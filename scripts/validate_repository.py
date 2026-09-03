"""Run the core offline checks for the frozen study repository."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    "scripts/self_test.py",
    "scripts/check_case_study_boms.py",
    "scripts/test_model_led_elcd_matching.py",
    "scripts/test_guardrails.py",
    "scripts/test_bounded_production.py",
    "scripts/test_four_class_reporting.py",
    "scripts/test_case_study_id_alignment.py",
    "scripts/test_dynamic_class4.py",
    "scripts/check_matching_protocol_identity.py",
    "scripts/check_qwen_benchmark_lock.py",
]


def main() -> None:
    subprocess.run([sys.executable, "-m", "compileall", "-q", "main.py", "production", "colab", "scripts"], cwd=ROOT, check=True)
    for rel in CHECKS:
        print(f"\n=== {rel} ===", flush=True)
        subprocess.run([sys.executable, rel], cwd=ROOT, check=True)
    print("\nRepository validation: PASS")


if __name__ == "__main__":
    main()
