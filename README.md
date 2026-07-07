# Leaving New York?

Code supporting the essay, "Leaving New York? Locations of US Literary Publishing Since 1945."

## Data

Two freely available external inputs are needed: the LOC MARC dataset (Step 1) and the
GeoNames gazetteer (Step 3). Fetch both with:

```bash
scripts/download_data.sh
```

or individually via `scripts/download_data.sh marc` / `scripts/download_data.sh geonames`.

The LOC download may not succeed automatically. If the script reports it
couldn't fetch the MARC dataset automatically, follow the printed manual
steps (download the dataset from
<https://lccn.loc.gov/2020445551> in a browser, save the zip where the
script tells you, then re-run it to extract).

## Pipeline

1. **Parse the raw MARC dump** (`imprints.data_collection`) into per-file
   pickles of record dicts — one dict per catalog record, keeping every
   field the rest of the pipeline needs.
2. **Clean and normalize** (`imprints.data_cleaning`) those pickles into a
   single flat CSV, `data/PS/data.csv`: filter to the LC class range under
   study, split multi-city publication fields into one row per city, and
   classify each place as `New York City` / `Other` / `No place of
   publication`.
3. **Geocode** each place of publication to a lat/lon (`imprints
   .marc_place_geonames`, `imprints.geonames_geocode`,
   `imprints.llm_geocode`, `imprints.join_geocoded`,
   `imprints.geocode_compare`) — mostly deterministically against
   a local gazetteer. An LLM cleans messy places of publication
   that can't be resolved deterministically (< 10% of US records).
4. **Plot** (`figures/scripts/*.py`) the cleaned/geocoded CSVs into the
   figures reported in the essay.
5. **Validate** (several smaller modules under `src/imprints/`) that the
   headline finding isn't an artifact of a specific modeling choice —
   how multi-place records are weighted, whether PS is unusual among LC
   classes, whether the NYC label agrees with an independently-derived one,
   etc.

Each step is detailed below, with the analytical decisions that shape it.

## Step 1: `data_collection`

Set `RAW_DATA_DIR` to wherever you store the `.xml.gz` files from the dataset downloaded above (`data/raw` if you used `scripts/download_data.sh`).

```bash
python -m imprints.data_collection \
    --input_dir RAW_DATA_DIR \
    --output_dir data \
    --class_range PS
```

This creates `data/PS` containing `.pkl` files with data from the `.xml.gz` files that we will then clean.

The MARC files are tens of gigabytes of XML; this step streams each record
with `lxml.iterparse` and discards it immediately after reading. Files are
processed in parallel, one worker per CPU core.

Key decisions:

- **Every record is kept, not just PS ones — matches are flagged, not
  filtered.** `--class_range PS` only sets a boolean (`matches_class_range`)
  on each record; every record, regardless of class, still ends up in the
  pickles. This is what lets later cross-class comparisons
  (`imprints.cross_range`, `imprints.class_totals`) ask "is this pattern
  unique to PS?" by re-reading the same pickles, instead of re-parsing the
  raw XML a second time.
- **Only *publication* fields count as a place/publisher, not production,
  distribution, or manufacture.** MARC's `264` field is typed by a second
  indicator distinguishing those four functions; a `260` field has no such
  indicator and is always treated as publication. Without this filter, a
  book's printer's city (a `264` with a different function code) could be
  mistaken for its place of publication. The `260`/`264` variants exist
  because cataloging practice changed over time — records use one or the
  other, never both consistently — so they're merged, order-preserving and
  deduplicated, into one `places`/`publishers`/`year` list per record.
- **Copyright/other dates are kept regardless of function.** A `264`'s date
  subfield is useful as a publication-year proxy even when its place
  isn't a place of publication (e.g. a copyright or manufacture field), so
  the date is pulled from every `260`/`264` occurrence while place/publisher
  are restricted to publication ones.
- **Extra structured signals are captured beyond the place-name text**, purely
  as independent columns that later steps *may* use for disambiguation or QA
  — none of them touch the core NYC/Other classification: the 008 control
  field, `044` country
  codes, the `752` hierarchical place name, subject/genre headings (`600`/`610`/`611`/`630`/`650`
  /`651`/`655`), local call numbers, and author birth/death dates. The last
  three exist specifically to support `imprints.secondary_classification`
  (see below), not the core place pipeline.

## Step 2: `data_cleaning`

Convert the pickles into a CSV with normalized places of publication.

```bash
python -m imprints.data_cleaning \
    --input_dir data/PS \
    --output_csv data/PS/data.csv \
    --class_range PS
```

This is where the record-per-book pickles become the row-per-place-of-
publication table (`data/PS/data.csv`) every downstream figure and analysis
reads by default.

Key decisions:

- **NYC identification is LLM-curated, not pattern-matched, and frozen into
  a reviewable file (`nyc_variants.txt`).** Values in `nyc_variants.txt` were
  identified by `gpt-5.2-2025-12-11` as referencing NYC and verified by hand from a complete list
  of unique cleaned placenames derived from the PS range. Fuzzy string
  matching (`"new york city"`, `"nyc"`, etc.) undercounts: publishers give
  their location at the neighborhood level (`"Flushing, NY"`) or even more
  specifically (`"Ditmas Ave."`), and an LLM reviewing the full list of
  distinct placenames catches those where a keyword search would not. Because
  the list is frozen rather than regenerated per run, editing what counts as
  NYC means editing that file, not the matching code — and every run is
  reproducible against the same definition.
- **A single publication field can list more than one city, and these are
  split so each is counted.** A `260`/`264` `$a` subfield often names
  co-publishing cities together, e.g. `"Boston and New York"` or
  `"New York; London"`. These are split on `;`, `&`, and the word `and` — but
  *not* inside parentheses/braces (an address) or inside a small set of
  country/place names that themselves contain "and" (`"Trinidad and
  Tobago"`, `"Bosnia and Herzegovina"`, etc.), which must stay intact.
  Skipping this step would silently drop the New York component of any
  co-published book from the count.
- **The table explodes on place, so row count isn't record count.** After
  splitting, the DataFrame is exploded so a book published in two cities
  becomes two rows, one per place. This is why the cleaned CSV can have more
  rows than there were input records, and why any per-record statistic
  (e.g. `imprints.multi_place_sensitivity`'s weighting checks) has to
  reaggregate by `lccn` rather than by row. Numeric columns that depend on
  the *whole record* (`class_digits`, `year_min`) are computed before the
  explode, precisely so they aren't accidentally recomputed per exploded row.
- **`year_min` is the earliest year found anywhere, including inside the
  publisher's own text.** Some records only carry a year embedded in the
  publisher subfield rather than in a dedicated date field; `year_min` takes
  whichever is earlier of the two, so a record isn't dropped just because
  its formal date subfield is missing.
- **Only publication-function fields feed place/publisher**, mirroring the
  same distinction made in `data_collection` (kept here too since this module
  can also run against pickles from other collection runs).
- **The 008/044/752 signals are decoded into readable columns
  (`place_name_008`, `country_names_044`, `place_752`) but deliberately do
  not feed `places`/`places_clean`/`city_group`.** They exist as
  cross-checks and as scoping signal for the geocoding step below, kept
  strictly separate from the label the essay's first two figures are built on.

## Additional place-of-publication signal (008, 044, 752)

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

## Step 3: Geocoding

Places of publication are geocoded in two passes, both against a local
GeoNames gazetteer — no external API calls. The primary pass matches
`places_clean` directly, scoped by `place_name_008` (e.g. a bare "athens" is
looked up only among Georgia places when `place_name_008` is "Georgia") —
free, deterministic, and, since `place_name_008` is present on ~94% of
PS-range records, enough to resolve ~90% of records outright. Only what that
pass can't resolve (no `place_name_008`, or no gazetteer match in scope)
falls through to an LLM-normalization pass, whose output is then matched
against the same gazetteer.

The reasoning behind the two-pass design: a plain gazetteer lookup is exact
string matching with no world knowledge — it can't tell that a cataloger's
"Santa Fe" almost certainly means New Mexico rather than Argentina, and it
can't disambiguate a bare "Athens" between Georgia, Ohio, Illinois, or
Greece. Bringing in an LLM for *every* record would be both wasteful and
less reliable than a deterministic match where one is available. Scoping by
`place_name_008` — a structured MARC code, not free text — resolves most of
that ambiguity for free, before an LLM is ever called; the LLM pass exists
specifically for the residual the structured signal can't scope on its own.

### GeoNames reference data

Downloaded into `data/geonames/` (not committed, like the raw MARC dump) by
`scripts/download_data.sh geonames`. Two different country lists are
involved, for two different reasons:

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

`imprints.marc_place_geonames` builds a small crosswalk (72 distinct
values in this corpus) from a decoded 008 place name (e.g. `"New York
(State)"`) to a GeoNames country/admin1 scope — reviewed once and frozen to
a CSV, the same "generate once, review, freeze" treatment as
`nyc_variants.txt`, rather than recomputed on every run.

`imprints.geonames_geocode` (`direct` mode) then groups the cleaned CSV by
unique (`places_clean`, `place_name_008`) pair, resolves each pair's scope
via that crosswalk, and matches `places_clean` against a scoped index of
GeoNames populated places. Key decisions inside the matcher:

- **Two separate name indices — a place's "primary" name and its
  "alternate" names — are consulted in that order, not merged.**
  GeoNames' `alternatenames` column mixes a place's genuine common short
  forms with stale historical and foreign-language names. Merging both
  tiers into one index let a former name outrank the real place with that
  name on a population tie-break (e.g. Lemont, IL was once called
  "Athens" and has a larger population than the real Athens, IL — merging
  the tiers would resolve "athens"/Illinois to the wrong town). Keeping
  them as two tiers, consulting alternate names only when primary has no
  candidate in scope, avoids that. Alternate names still matter: GeoNames'
  own primary name for New York City is literally "New York City", so the
  bare "New York" almost every cataloger actually writes only matches via
  the alternate-name tier.
- **A tie within a scope is broken by population and flagged, not left
  unmatched.** Multiple identically-named places can appear within one
  scope; rather than give up, the higher-population candidate is chosen
  and the row is marked `geonames_ambiguous` so this remains visible for
  quality review instead of silently picking a possibly-wrong candidate.
- **A failed exact match gets one retry against a canonicalized form** that
  strips a redundant trailing state token (e.g. `"waterville me"` retries
  as `"waterville"`), since GeoNames indexes the bare city name, not the
  cataloger's "city + state abbreviation" convention.

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

`imprints.llm_geocode` asks an LLM to turn a possibly abbreviated,
inconsistently punctuated, or mistyped place string into a normalized
`"city, state"` / `"city, country"` guess. Key decisions:

- **The model is told to return no guess rather than a fabricated one**
  when the input is too garbled or generic ("various places", "s.l.",
  "n.p.") to correspond to a real, identifiable place — a wrong guess would
  silently corrupt the map, whereas a `null` just leaves that row
  unresolved and visible as such.
- **It's told to default to a US reading of an ambiguous city name** (since
  the whole corpus is American literary publishing) *unless* the 008/752
  MARC signal, when present, says otherwise — that structured signal is
  explicitly weighted above the model's own generic heuristic, since it's
  cataloger-supplied ground truth rather than a guess from the city name
  alone.
- **The run is resumable and fault-tolerant**: results are appended to the
  output CSV row by row as they're produced, already-processed rows are
  skipped on restart, and a single failed request is skipped (not fatal) so
  a large batch run isn't lost to one transient API error.

`imprints.join_geocoded` then coalesces both pathways: a `geo_key` the
direct pass resolved wins over the residual pass's result for it, with a
`geocode_source` column recording which pathway produced each row.

```bash
python -m imprints.join_geocoded
```

Both the geocoding passes and this join key on `geo_key` —
`places_clean` combined with `place_name_008`, not `places_clean` alone.
This is deliberate: grouping on the bare cleaned string alone would merge
"Athens, GA" and "Athens, OH" into a single group before either pass ever
gets a chance to distinguish them, since MARC's 008 code is exactly the
signal that keeps them apart. `imprints.place_keys` is the single shared
module every producer/consumer of this key uses, so the key is always
built identically.

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

## Step 4: Visualizations and reported figures

Scripts to create plots and figures reported inline live in `figures/scripts/`. Regenerate every figure that needs only `data/PS/data.csv` at once:

```bash
python figures/scripts/make_figures.py
```

or run an individual figure:

```bash
python figures/scripts/ps_nyc_share.py
```

By default, they assume that you have placed the outputs of `data_collection` and `data_cleaning` in `data/PS/data.csv`. Two figures need more than that CSV and are therefore *not* run by `make_figures.py`: `fig1_primary_only.py` (needs `data/PS/secondary_classification.csv`) and `nyc_peak_map_simple.py` (needs `data/PS/geocoded.csv` plus `cartopy`); run them individually once their inputs exist.

Generated figures are written to `figures/outputs/` by default, each as PNG, SVG, and PDF. Shared plotting defaults (Helvetica Now Micro font, 1200 DPI, the Okabe-Ito colorblind-safe palette with marker/linestyle cycling so series stay separable in grayscale print, output formats) live in `figures/scripts/style.py`.

Smoothing defaults differ deliberately by figure: the headline share and count figures (`ps_nyc_share`, `fig1_primary_only`) plot raw annual values, while the noisier many-series and first-appearance figures (`ps_range_nyc_share`, `new_publisher_nyc_share`) apply a 5-year rolling mean by default. Every script exposes `--smooth`/`--no-smooth` to override.

What each figure script does and computes:

- **`ps_nyc_share.py`** — NYC's share of PS publications each year:
  `NYC / (NYC + Other)`, excluding records with no identifiable place of
  publication (so the denominator is only records that *do* say where they
  were published). Also reports the year the share first/last crossed 50%
  and its peak year — the specific claims the essay makes about the arc of
  NYC's dominance.
- **`ps_nyc_counts.py`** — the same NYC/Other split as absolute annual
  counts rather than a share, optionally including the "no place of
  publication" series on its own — useful for seeing whether a falling
  share reflects NYC output actually shrinking or other places' output
  growing around it.
- **`ps_unique_publishers.py`** — count of *distinct* cleaned publisher
  names per year, a proxy for how consolidated or fragmented the publishing
  industry is in a given year.
- **`nyc_and_publishers.py`** — combines the previous two into one two-panel
  figure and, if asked, reports the correlation between the "Other" series
  and the unique-publisher count — checking whether the growth of
  publishing outside NYC tracks the growth in the number of distinct
  publishers.
- **`ps_range_nyc_share.py`** — the NYC-share line of `ps_nyc_share` computed
  separately for each large PS sub-range (see `imprints.ps_ranges`), backing
  the claim that the post-peak decline is not driven by any one sub-range.
- **`new_publisher_nyc_share.py`** — NYC's share of publisher-place pairings
  appearing in PS for the *first time* each year, plus the pooled
  before/after-peak odds comparison cited in the essay ("the odds of a new
  PS publisher having an NYC imprint fall to less than half").
- **`fig1_primary_only.py`** — not run by `make_figures.py` (it needs
  `data/PS/secondary_classification.csv`). Re-plots the `ps_nyc_share` line
  with secondary literature (criticism/scholarship/reference, per
  `imprints.secondary_classification`) excluded, backing the footnote that
  excluding it changes no core finding.
- **`nyc_peak_map_simple.py`** — not run by `make_figures.py` (it needs
  `data/PS/geocoded.csv`, i.e. the geocoding step above, plus `cartopy`, a
  heavier optional dependency). Maps every geocoded US coordinate before vs.
  after NYC's peak-share year, with point color and size both on one shared
  scale across both panels — a shared scale specifically so that "output
  elsewhere grew" (rather than "NYC's own output shrank") reads directly off
  the map instead of requiring two independently-scaled panels to be
  compared by eye.

## Supplementary validation analyses

These modules live under `src/imprints/` alongside the core pipeline but
don't feed `data/PS/data.csv` or the figures above. Each exists to check
whether the headline NYC finding is robust to a specific modeling choice,
rather than an artifact of it.

- **`imprints.multi_place_sensitivity`** — the cleaned CSV gives every place
  on a multi-place record equal weight (a book published in both NYC and
  Boston counts once for each). This script recomputes the NYC-share series
  under two alternative treatments — dropping multi-place records entirely,
  or splitting each one's weight fractionally across its places — and
  reports whether the reported crossing/peak years change under any of the
  three.
- **`imprints.cross_range` / `imprints.class_totals`** — the essay's finding
  is about class `PS` specifically. `cross_range` rebuilds the same
  NYC-share time series for *every* LC class/subclass (reusing the raw
  pickles, which retain all classes, not just PS) to check whether the
  rise-then-fall pattern is unique to American literature or shared broadly
  across LC. `class_totals` reports simple record counts for class `P` vs.
  subclass `PS` as a sanity check on scope.
- **`imprints.compare_nyc_identification`** — cross-checks the primary NYC
  label (LLM-curated match against `260`/`264` place text) against an
  independently-built label using `752`'s hierarchical `$d` (city) subfield
  — a different MARC field, decoded by a different method — and reports
  where the two agree and disagree, at the record level.
- **`imprints.secondary_classification`** — classifies each PS record as a
  primary literary work (a novel, a poetry collection) vs. secondary
  literature (criticism, scholarship, reference works about a work or
  author), using rules over subject headings, the 008 literary-form byte,
  and self-reference detection (so an author's own memoir isn't
  miscategorized as criticism about them). Used to check that raw
  publication counts aren't substantially inflated by scholarship about
  literature rather than literature itself.
- **`imprints.first_appearance_stats` / `imprints.get_unique_places` /
  `imprints.get_ps_places`** — raw `places_clean` strings overcount distinct
  physical places, since the same city appears under several
  cataloging-convention spellings (`"boston"`, `"boston ma"`, `"boston
  mass"`). These scripts canonicalize state-name variants
  (`imprints.place_canonicalization`) before counting, to support claims
  about how many genuinely distinct places of publication appear, and when.
- **`imprints.ps_ranges`** — groups PS's fine-grained LC numbers (e.g.
  `PS3573`) into named sub-ranges keyed by an individual author's career era
  (colonial, 19th century, 1900–1960, 1961–2000, 2001–), the single source
  of truth for any figure or analysis comparing the NYC story *within* PS
  rather than across all of it.
- **`imprints.repeats`** — reports what percentage of distinct LCCNs
  (Library of Congress Control Numbers) appear more than once in the
  dataset, a basic duplication check on the source data.

## AI Statement

I used LLMs by Anthropic and OpenAI to help write, refactor, and document code.
