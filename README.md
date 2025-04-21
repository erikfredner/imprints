# Imprints

Tools for extracting and analyzing publication imprints from the Library of Congress MARC XML dataset across different LC classification ranges (e.g., PS, PZ, F, E), and visualizing trends in publication locations and metadata.

## Prerequisites

- Python 3.12 or higher
- Git
- Approximately 20 GB of disk space for raw MARC XML dumps and intermediate files
- Unix-like OS (macOS/Linux)
- `uv`

## Download Raw Data

Download the *Library of Congress Books All MARC Records* (2019) dataset from:
<https://lccn.loc.gov/2020445551>

Unpack the archive so that `RAW_DATA_DIR` contains your `.xml.gz` files.

## 1. Data Collection by LC Class

Extract records for each LC class of interest (e.g., PS):

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

## 2. Data Cleaning

Convert pickles into cleaned, flat CSVs with expanded publication places:

```bash
python -m imprints.data_cleaning \
    --input_dir data/PS \
    --output_csv data/PS/data.csv \
    --class_range PS

# Repeat for PZ, F, E as needed
```

Optionally, also save a cleaned DataFrame pickle:

```bash
python -m imprints.data_cleaning \
    --input_dir data/PS \
    --output_csv data/PS/data.csv \
    --output_pkl data/PS/data_cleaned.pkl \
    --class_range PS
```

## 3. New York Publication Locations (MDS)

Generate a CSV of all publication rows flagged by New York location:

```bash
python -m imprints.data_collection_ny \
    --input_dir RAW_DATA_DIR \
    --output_csv data/MDS_pub_locations.csv
```

## 4. Visualization

All plotting scripts are in the `viz/` directory. They read from `data/` and write PNG figures to `viz/`.

Basic usage examples:

```bash
# PS imprint counts (New York vs Other)
python viz/plot_ps_counts.py --input-csv data/PS/data.csv

# PS New York share
python viz/plot_ps_new_york_share.py --input-csv data/PS/data.csv
```
