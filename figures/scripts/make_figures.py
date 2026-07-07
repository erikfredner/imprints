#!/usr/bin/env python3
"""Regenerate every figure that needs only data/PS/data.csv (ps_nyc_share,
ps_nyc_counts, ps_unique_publishers, nyc_and_publishers, ps_range_nyc_share,
and new_publisher_nyc_share) into figures/outputs/.

Two figure scripts are deliberately not run here because they need inputs
beyond data.csv: fig1_primary_only.py (needs
data/PS/secondary_classification.csv from imprints.secondary_classification)
and nyc_peak_map_simple.py (needs data/PS/geocoded.csv from the geocoding
step, plus cartopy). Run those individually once their inputs exist.

Runs each figure script as a subprocess so each keeps its own argparse defaults
and an isolated matplotlib state. Exits non-zero if any figure fails.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "ps_nyc_share.py",
    "ps_nyc_counts.py",
    "ps_unique_publishers.py",
    "nyc_and_publishers.py",
    "ps_range_nyc_share.py",
    "new_publisher_nyc_share.py",
]


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
