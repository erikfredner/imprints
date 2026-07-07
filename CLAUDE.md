# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code supporting the essay "Leaving New York? Locations of US Literary Publishing Since 1945." It processes the Library of Congress's "Books (All)" MARC records (2019) to track where US literary works (LC class `PS`) were published over time, with a focus on the New-York-City vs. elsewhere split. The end products are the figures in `figures/outputs/`.

The input dataset is **not** in the repo. Download the `.xml.gz` MARC files from <https://lccn.loc.gov/2020445551> and point the collection step at them.

## Environment & tooling

Managed with `uv` (Python ≥3.12; `.python-version` pins 3.14). The `imprints` package lives in `src/imprints/` and is run as modules from the repo root.

```bash
uv sync                 # install deps + dev (ruff, pytest)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pytest           # tests (fast; no data files needed)
```

There is no build step and no CI; run the linter and tests locally. Prefix any run command with `uv run` (e.g. `uv run python -m imprints.data_cleaning ...`).

## Pipeline (run in order)

1. **`imprints.data_collection`** — parse raw MARC `.xml.gz` → per-file `.pkl` lists of record dicts. Streams records with `lxml` `iterparse` and fans out across CPUs with `ProcessPoolExecutor`. Writes to `data/<class_range>/`. Keeps *all* records but flags those matching `--class_range`.
2. **`imprints.data_cleaning`** — load all pickles → filter to the class range → one flat CSV. This is where normalization happens (see below). Output: `data/PS/data.csv`, the input every figure script expects by default.
3. **`figures/scripts/*.py`** (descriptively named: `ps_nyc_share.py`, `nyc_and_publishers.py`, …) — read the cleaned CSV and emit `figures/outputs/<name>.{png,svg,pdf}`. Each is a standalone `argparse` script defaulting to `data/PS/data.csv` in / `figures/outputs/<name>.png` out (the SVG/PDF siblings are derived from the stem). `figures/scripts/make_figures.py` regenerates every figure that needs only `data.csv`; `fig1_primary_only.py` (needs `secondary_classification.csv`) and `nyc_peak_map_simple.py` (needs `geocoded.csv` + cartopy) run individually. Shared plot defaults (font, 1200 DPI, Okabe-Ito colors with marker/linestyle cycling, multi-format save) live in `figures/scripts/style.py`.

Helper modules: `imprints.get_unique_places` (dump sorted unique `places_clean`, used to build the NYC variant list) and `imprints.repeats` (% of LCCNs appearing more than once).

See `README.md` for the exact invocation flags of steps 1–2.

## Architecture notes that aren't obvious from one file

- **Class-range parsing is duplicated** in `data_collection.py` and `data_cleaning.py` (`parse_range_spec`, `parse_class`, `matches_range`). A `class_range` is a dict `{"prefix", "min", "max"}` parsed from strings like `PS` or `PR9000-PR9999`. Changes to range semantics must be made in both modules.
- **MARC field mapping** (in `process_record`): `050$a`=LC classification, `010$a`=LCCN, `100$a`=author, `245$a`=title, `260`/`264` subfields `a`=place, `b`=publisher, `c`=year. The `260` and `264` variants are merged (dedup, order-preserving) because records use one or the other.
- **NYC normalization is LLM-curated, not algorithmic.** `nyc_variants.txt` is a frozen list of cleaned placenames an LLM judged to refer to NYC (neighborhoods, street-level imprints, misspellings — things fuzzy string matching missed). `get_target_cities` classifies each cleaned place as `New York City` / `Other` / `No place of publication` purely by membership in this set. Editing the NYC definition means editing that file, not the code. `clean_string` (lowercase, strip non-letters, collapse spaces) must be applied consistently so CSV values and `nyc_variants.txt` entries compare equal.
- **The cleaning pipeline explodes on `places`**, so one source record can become multiple rows — row counts after cleaning can exceed the input record count. Numeric columns (`class_digits`, `year_min`) are computed before the explode. `year_min` is the min of years found in `260/264$c` and in the publisher subfield. Rows missing `class_digits` or `year_min` are dropped.
