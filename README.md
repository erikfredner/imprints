# Leaving New York?

Code supporting the essay, "Leaving New York? Locations of US Literary Publishing Since 1945."

## Data

Download the Library of Congress's Books (All) MARC Records (2019) dataset from here:

<https://lccn.loc.gov/2020445551>

## `data_collection`

Set `RAW_DATA_DIR` to wherever you store the `.xml.gz` files from the dataset downloaded above.

```bash
python -m imprints.data_collection \
    --input_dir RAW_DATA_DIR \
    --output_dir data \
    --class_range PS
```

This creates `data/PS` containing `.pkl` files with data from the `.xml.gz` files that we will then clean.

## `data_cleaning`

Convert the pickles into a CSV with normalized places of publication.

Values in `data/nyc_variants.txt` were identified by `gpt-5.2-2025-12-11` as referencing NYC from a complete list of unique cleaned placenames derived from the PS range. Due to errors and inconsistencies in the spelling and representation of placenames (both by publishers and by LC catalogers), using an LLM to identify placenames associated with NYC produced fewer errors than programmatic searches for fuzzy matching strings like, `"new york city"`, `"nyc"`, etc. Even though matching on major placenames (such as adding `"brooklyn"` to the above list) would seem to be adequate, in practice it missed some places of publication given at the neighborhood level within NYC (e.g., `"Flushing, NY"`) or even more specific places of publication (e.g., `"Ditmas Ave."`)

```bash
python -m imprints.data_cleaning \
    --input_dir data/PS \
    --output_csv data/PS/data.csv \
    --class_range PS
```

## Visualizations and reported figures

Scripts to create plots and figures reported inline are in the `viz/` directory, and can be regenerated like so:

```bash
python viz/fig1.py
```

By default, they assume that you have placed the outputs of `data_collection` and `data_cleaning` in `data/PS/data.csv`

`predict.py` creates the linear model referenced inline, as well as a figure that is not included in the article.

## AI Statement

I used [OpenAI's `codex`](https://github.com/openai/codex) to help write and refactor code.
