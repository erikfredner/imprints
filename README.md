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

Values in `nyc_variants.txt` were identified by `gpt-5.2-2025-12-11` as referencing NYC from a complete list of unique cleaned placenames derived from the PS range. Due to errors and inconsistencies in the spelling and representation of placenames (both by publishers and by LC catalogers), using an LLM to identify placenames associated with NYC produced fewer errors than programmatic searches for fuzzy matching strings like, `"new york city"`, `"nyc"`, etc. Even though matching on major placenames (such as adding `"brooklyn"` to the above list) would seem to be adequate, in practice it missed some places of publication given at the neighborhood level within NYC (e.g., `"Flushing, NY"`) or even more specific places of publication (e.g., `"Ditmas Ave."`)

```bash
python -m imprints.data_cleaning \
    --input_dir data/PS \
    --output_csv data/PS/data.csv \
    --class_range PS
```

Two normalization choices about *place of publication* are worth noting:

- **Only publication places are counted.** MARC `264` fields are typed by their
  second indicator (production, publication, distribution, manufacture,
  copyright). `data_collection` keeps the place (`$a`) and publisher (`$b`)
  only from publication fields (`264` with indicator blank or `1`, and all of
  `260`), so a printer's or distributor's city is not mistaken for a place of
  publication. Copyright/other dates (`$c`) are still used when estimating the
  publication year.
- **Compound places are split.** A single subfield often lists co-publication
  cities together (e.g. `"Boston and New York"`). `data_cleaning` splits these
  on `;`, `&`, and the word `and` so that a New York component is recognized.

### Additional place-of-publication signal (008, 044, 752)

Beyond the `260`/`264` `$a` text that drives `places`/`places_clean`/
`city_group` above, the pipeline also captures three other MARC signals of
place of publication, purely as extra columns for downstream use (e.g.
geocoding cross-checks or filling gaps where `260`/`264` is missing or
vague) — **none of these feed the NYC/Other classification**:

- `place_code_008` / `place_name_008` — the coded place of publication at
  008 bytes 15-17, decoded to a name.
- `country_codes_044` / `country_names_044` — additional MARC country codes
  from `044 $a`, decoded to names; `country_codes_044_iso` keeps `044 $c`
  ISO codes raw (a different code system, not decoded here).
- `place_hierarchy_752` / `place_752` — the hierarchical place name from
  `752` (country › state/province › county › city › city subsection), kept
  raw (structured) and flattened to a human-readable string.

The 008/044 codes are decoded via `marc_country_codes.csv` at the repo
root, generated from the Library of Congress's official "MARC List of
Countries" (<https://www.loc.gov/standards/codelists/countries.xml>). It
includes obsolete/historical codes (e.g. Soviet Union, Czechoslovakia,
Yugoslavia) alongside current ones, since these MARC records span
1945–2019 and older records may carry a code that was current at their own
cataloging date.

## Geocoding

Places of publication are geocoded in two passes, both against a local
GeoNames gazetteer — no external API calls. The primary pass matches
`places_clean` directly, scoped by `place_name_008` (e.g. a bare "athens" is
looked up only among Georgia places when `place_name_008` is "Georgia") —
free, deterministic, and, since `place_name_008` is present on ~94% of
PS-range records, enough to resolve ~90% of records outright. Only what that
pass can't resolve (no `place_name_008`, or no gazetteer match in scope)
falls through to an LLM-normalization pass, whose output is then matched
against the same gazetteer.

### GeoNames reference data

Download into `data/geonames/` (not committed, like the raw MARC dump).
Two different country lists are involved, for two different reasons:

```bash
mkdir -p data/geonames && cd data/geonames
curl -fsSL -o admin1CodesASCII.txt https://download.geonames.org/export/dump/admin1CodesASCII.txt
for cc in US CA GB AU \
          DE IT FR IE IN JP ES CN CH RU MX NL SE PL FI RO BR IL DK CU ZA NG \
          BE AR PR TW UA AT NZ LB KR CZ NO PT PH GR VI EG TR EC IR BA CL; do
    curl -fsSL -o "${cc}.zip" "https://download.geonames.org/export/dump/${cc}.zip"
    unzip -o "${cc}.zip" "${cc}.txt"
    rm "${cc}.zip"
done
```

- **`US CA GB AU`** are the only countries the *direct* pass needs: every
  distinct `place_name_008` value in the PS-range corpus is a US state/DC, a
  Canadian province, a UK constituent country, or one of a few Australian
  states (see `imprints.marc_place_geonames`'s docstring). If a future
  corpus carries a `place_name_008` value from a country not yet
  downloaded, the crosswalk-building step below will print it as unresolved
  by name, so you know which additional country file to fetch.
- The rest are for the *llm* pass only: countries `place_name_008` never
  names but that `imprints.llm_geocode`'s normalized answers do, for
  records the direct pass can't scope at all. This list was generated by
  checking which LLM answers were going unmatched purely for lack of a
  country entry, then keeping every country appearing >=5 times in that
  residual (see `imprints.marc_place_geonames._COUNTRY_NAME_TO_CODE_CI`'s
  comment) — reviewed once and frozen, not derived fresh each run. It's
  fine to extend `_COUNTRY_NAME_TO_CODE_CI` and this list further if a
  future run's llm-mode residual turns up other countries worth adding.

### Building the crosswalk and running the direct pass

```bash
# One-time (or after the place_name_008 value set changes): freezes
# marc_place_008_geonames.csv at the repo root.
python -m imprints.marc_place_geonames \
    --data_csv data/PS/data.csv \
    --admin1_codes data/geonames/admin1CodesASCII.txt \
    --output_csv marc_place_008_geonames.csv

# Direct GeoNames match -> data/PS/geonames_direct.csv
python -m imprints.geonames_geocode direct \
    --input_csv data/PS/data.csv \
    --output_csv data/PS/geonames_direct.csv
```

### Fallback pipeline and reporting

`imprints.llm_geocode` only needs to run for what `geonames_direct.csv`
didn't resolve — pass it to `--skip_geo_keys_csv` to skip already-resolved
`geo_key`s — and its LLM-normalized output is then matched against the same
GeoNames gazetteer via `imprints.geonames_geocode`'s `llm` mode:

```bash
python -m imprints.llm_geocode \
    --skip_geo_keys_csv data/PS/geonames_direct.csv

python -m imprints.geonames_geocode llm \
    --input_csv data/PS/llm_geocode.csv \
    --output_csv data/PS/geonames_llm.csv
```

`imprints.join_geocoded` then coalesces both pathways: a `geo_key` the
direct pass resolved wins over the residual pass's result for it, with a
`geocode_source` column recording which pathway produced each row.

```bash
python -m imprints.join_geocoded
```

`imprints.geocode_compare` reports match-rate diagnostics: direct-pathway
coverage, and a disambiguation check confirming that ambiguous bare city
names (e.g. "athens" as Georgia/Ohio/Illinois, "columbia" as Missouri/South
Carolina) resolve to distinct, correct places.

```bash
python -m imprints.geocode_compare
```

## Tests

```bash
uv run pytest
```

## Visualizations and reported figures

Scripts to create plots and figures reported inline live in `figures/scripts/`. Regenerate them all at once:

```bash
python figures/scripts/make_figures.py
```

or run an individual figure:

```bash
python figures/scripts/ps_nyc_share.py
```

By default, they assume that you have placed the outputs of `data_collection` and `data_cleaning` in `data/PS/data.csv`

Generated figures are written to `figures/outputs/` by default, each as PNG, SVG, and PDF. Shared plotting defaults (Helvetica Now Micro font, 1200 DPI, grayscale style, output formats) live in `figures/scripts/style.py`.

`predict.py` creates the linear model referenced inline, as well as a figure that is not included in the article.

`city_growth.py` asks whether NYC's *falling share* after its ~1958 peak is
mostly a denominator effect — everywhere else growing rather than NYC shrinking.
It ranks and plots (raw counts vs. year, default 1950–2010) the top publishing
locations *outside* NYC.

```bash
python figures/scripts/city_growth.py   # top non-NYC cities by record count
```

It writes **two** variants: `city_growth.png` (all publishers) and
`city_growth_no_large_print.png`. The raw counts are dominated by a handful of
large-print/reprint houses clustered in Waterville/Thorndike, ME (Thorndike
Press, G.K. Hall, Wheeler, Center Point, Five Star, …), which reprint existing
titles rather than originate literary publishing; the second variant drops them
(see `LARGE_PRINT_MARKERS`) so the underlying trend in literary publishing is
legible.

Place strings encode the state inconsistently (`louisville ky`, `calif`,
`garden city n y`, `ohio`, or no state at all for foreign cities like `london`),
so the script splits city from state with a curated token map (`STATE_TOKENS`)
and folds obvious typos into their canonical spelling via string similarity
(`difflib`). Both steps are first-pass heuristics: the run prints the fuzzy
merges (with scores) and the most common *unrecognized* trailing tokens so the
map and threshold can be refined, and it flags city names that are really
several different places (e.g. Portland OR vs. ME) that each grow. The full
rankings are also written to `figures/outputs/city_growth*_ranking.csv`.

## PS broken down by numerical sub-range

Three figures (`ps_range_nyc_share`, `fig6`, `fig7`) break PS apart by its Library
of Congress numerical sub-ranges (the "individual authors by period" ranges plus
the genre/collection ranges) to ask whether the overall rise-then-fall story is
uniform across PS, and whether the post-peak decline is a *within-range* effect
or a *composition* effect. They read `data/PS/data.csv` directly (no extra build
step) and bin records with the shared mapping in `imprints.ps_ranges`; the
per-range NYC share is computed among placed imprints (`NYC / (NYC + Other)`),
like the cross-range analysis.

```bash
python figures/scripts/ps_range_nyc_share.py   # NYC share over time for the largest sub-ranges
python figures/scripts/fig6.py                 # composition of PS over time (two area charts)
python figures/scripts/fig7.py                 # how much of the decline is composition?
```

- **`ps_range_nyc_share`** — one NYC-share line per sub-range with at least
  10,000 records (PS700-893, "Individual authors · Colonial", falls well short
  of that and is dropped), ordered by PS range in a legend below the plot with
  each range's record count (`N`).
- **`fig6`** — two stacked-area charts, `fig6_counts` (absolute records per range)
  and `fig6_share` (the same normalized to 100% per year).
- **`fig7`** — a shift-share decomposition of the peak-to-later change into
  within / composition / interaction terms (with per-range composition
  contributions and a confirming OLS) written to `fig7_decomposition.csv`, plus a
  counterfactual figure comparing the actual NYC share to composition-frozen and
  within-frozen trajectories.

## Cross-range comparison

To ask whether PS's New-York-share pattern is particular to PS, general to LC, or
shared by a subset of ranges, the analysis is re-run for *every* LC range at both
the top level (single letter, e.g. `P`) and the subclass level (alpha prefix, e.g.
`PS`, `PR`). The `data/PS` pickles retain *all* records from the raw MARC XML, so no
re-collection is needed — `imprints.cross_range` streams them once and writes small
count tables.

```bash
# 1. Build NYC/Other count tables for all ranges -> data/cross_range/counts_*.csv
python -m imprints.cross_range --input_dir data/PS --output_dir data/cross_range

# 2. Per-range statistics + comparison figures -> figures/outputs/
python figures/scripts/cross_range.py
```

Step 2 writes `cross_range_stats.csv` (per-range peak year, rise/fall slopes,
humpiness percentile, and correlation with PS) and five figures:
`cross_range_subclass`, `cross_range_top`, `ps_vs_rest`,
`cross_range_small_multiples`, and `cross_range_crossing50` (only the subclasses
whose NYC share ever reaches 50%, each drawn with a distinct linestyle/marker). The NYC share here is computed among imprints with an
identifiable place (`NYC / (NYC + Other)`), and a record counts toward every range
its classifications match.

## AI Statement

I used [OpenAI's `codex`](https://github.com/openai/codex) to help write and refactor code.
