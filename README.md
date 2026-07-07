# US Literature in NYC

[![DOI](https://zenodo.org/badge/966983150.svg)](https://doi.org/10.5281/zenodo.21245095)

Code supporting our essay about publishing US literature in NYC.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
uv sync
```

## Data

Two freely available external inputs are needed: the LOC MARC "Books (All)"
dataset and the GeoNames gazetteer. Fetch both with:

```bash
scripts/download_data.sh
```

(or individually: `scripts/download_data.sh marc` / `scripts/download_data.sh geonames`).
If the MARC download can't complete automatically, the script prints manual
steps: download <https://lccn.loc.gov/2020445551> in a browser, save the zip
where the script says, and re-run it to extract.

## Step 1: Parse the raw MARC dump

```bash
uv run python -m imprints.data_collection \
    --input_dir data/raw \
    --output_dir data \
    --class_range PS
```

Writes per-file pickles of record dicts to `data/PS/`. (`--input_dir` is
wherever the `.xml.gz` files live; `data/raw` if you used the download script.)

## Step 2: Clean and normalize

```bash
uv run python -m imprints.data_cleaning \
    --input_dir data/PS \
    --output_csv data/PS/data.csv \
    --class_range PS
```

Produces `data/PS/data.csv` — one row per place of publication, with each
place classified as `New York City` / `Other` / `No place of publication`
(NYC membership is defined by `nyc_variants.txt`).
This CSV is the default input for every figure and analysis below.

## Step 3: Geocoding (needed only for the map figure)

```bash
# One-time: freeze the 008-place -> GeoNames scope crosswalk.
uv run python -m imprints.marc_place_geonames \
    --data_csv data/PS/data.csv \
    --admin1_codes data/geonames/admin1CodesASCII.txt \
    --output_csv marc_place_008_geonames.csv

# Direct gazetteer match (resolves ~90% of records).
uv run python -m imprints.geonames_geocode direct \
    --input_csv data/PS/data.csv \
    --output_csv data/PS/geonames_direct.csv

# LLM fallback for the unresolved residual, then match it too.
uv run python -m imprints.llm_geocode \
    --skip_geo_keys_csv data/PS/geonames_direct.csv
uv run python -m imprints.geonames_geocode llm \
    --input_csv data/PS/llm_geocode.csv \
    --output_csv data/PS/geonames_llm.csv

# Coalesce both passes into data/PS/geocoded.csv.
uv run python -m imprints.join_geocoded

# Optional: match-rate and disambiguation diagnostics.
uv run python -m imprints.geocode_compare
```

## Step 4: Figures

Regenerate every figure that needs only `data/PS/data.csv`:

```bash
uv run python figures/scripts/make_figures.py
```

Two figures need extra inputs and run individually:

```bash
# Needs data/PS/secondary_classification.csv
# (from: uv run python -m imprints.secondary_classification)
uv run python figures/scripts/fig1_primary_only.py

# Needs data/PS/geocoded.csv (Step 3) and cartopy.
uv run python figures/scripts/nyc_peak_map_simple.py
```

Figures are written to `figures/outputs/` as PNG, SVG, and PDF.

## Supplementary validation analyses

Each checks that the headline NYC finding is robust to a specific modeling
choice; all read `data/PS/data.csv` and/or the Step 1 pickles:

```bash
uv run python -m imprints.multi_place_sensitivity   # multi-place record weighting
uv run python -m imprints.cross_range               # is the pattern unique to PS?
uv run python -m imprints.class_totals              # P vs. PS record counts
uv run python -m imprints.compare_nyc_identification # NYC label vs. 752-derived label
uv run python -m imprints.secondary_classification  # primary vs. secondary literature
uv run python -m imprints.first_appearance_stats    # distinct-place counts over time
uv run python -m imprints.repeats                   # LCCN duplication check
```

## Tests

```bash
uv run pytest
```

## AI Statement

I used LLMs by Anthropic and OpenAI to help write, refactor, and document code.
