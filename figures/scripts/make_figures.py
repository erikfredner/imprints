#!/usr/bin/env python3
"""Regenerate every figure (fig1-fig4 and predict) into figures/outputs/.

Runs each figure script as a subprocess so each keeps its own argparse defaults
and an isolated matplotlib state. Exits non-zero if any figure fails.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPTS = ["fig1.py", "fig2.py", "fig3.py", "fig4.py", "predict.py"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all imprints figures")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional cleaned PS data CSV forwarded to every figure script",
    )
    args = parser.parse_args()

    passthrough = []
    if args.input_csv is not None:
        passthrough = ["--input-csv", str(args.input_csv)]

    for name in SCRIPTS:
        script = SCRIPTS_DIR / name
        print(f"\n=== {name} ===")
        subprocess.run([sys.executable, str(script), *passthrough], check=True)

    print("\nAll figures regenerated.")


if __name__ == "__main__":
    main()
