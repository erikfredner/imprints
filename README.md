# Imprints

Tools for extracting and analyzing publication imprints from the Library of Congress MARC XML dataset across different LC classification ranges (e.g., PS, PZ, F, E), and visualizing trends in publication locations and metadata.

## Download

Download the Library of Congress's Books (All) MARC Records (2019) dataset from:
<https://lccn.loc.gov/2020445551>

## `data_collection`

```bash
# Example: PS class
python -m imprints.data_collection \
    --input_dir RAW_DATA_DIR \
    --output_dir data \
    --class_range PS

# Repeat for other classes:
python -m imprints.data_collection --input_dir RAW_DATA_DIR --output_dir data --class_range PZ
python -m imprints.data_collection --input_dir RAW_DATA_DIR --output_dir data --class_range F
python -m imprints.data_collection --input_dir RAW_DATA_DIR --output_dir data --class_range E
```

This creates `data/PS`, `data/PZ`, etc., each containing `.pkl` pickle files.

## `data_cleaning`

Convert pickles into cleaned, flat CSVs with normalized places of publication.

Values in `data/nyc_variants.txt` are those identified by `gpt-5.2-2025-12-11` as referencing NYC from a complete list of unique cleaned placenames derived from the PS range. Due to errors and inconsistencies in the spelling and representation of placenames (both by publishers and by LC catalogers), using an LLM produced fewer false negatives than programmatic searches for fuzzy matching key strings like e.g., `"new york city"`, `"nyc"`, etc.

```bash
python -m imprints.data_cleaning \
    --input_dir data/PS \
    --output_csv data/PS/data.csv \
    --class_range PS
```

## Visualization

All plotting scripts are in the `viz/` directory. They read from `data/` and write PNG figures to `viz/`.

Basic usage examples:

```bash
# PS imprint counts (New York vs Other)
python viz/plot_ps_counts.py --input-csv data/PS/data.csv

# PS New York share
python viz/plot_ps_new_york_share.py --input-csv data/PS/data.csv
```
