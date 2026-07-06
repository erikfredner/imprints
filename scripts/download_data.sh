#!/usr/bin/env bash
# Downloads the two external inputs the pipeline needs and places them
# where imprints.data_collection / imprints.marc_place_geonames expect:
#   - LOC "Books (All)" MARC 2019 dataset -> data/raw/*.xml.gz
#   - GeoNames per-country gazetteer files -> data/geonames/*.txt
#
# Usage:
#   scripts/download_data.sh [marc|geonames]
#   (no argument runs both)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
GEO_DIR="$ROOT_DIR/data/geonames"

MARC_ITEM_URL="https://hdl.loc.gov/loc.gdc/gdcdatasets.2020445551_2019"
MARC_ZIP="$RAW_DIR/Books.All.2019.zip"

# A browser UA, since loc.gov's Cloudflare front-end blocks the default
# curl UA outright before even getting to the bot challenge.
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

manual_marc_instructions() {
    cat <<EOF

Could not fetch the MARC dataset automatically. $MARC_ITEM_URL is a
landing page behind a Cloudflare bot challenge, not a direct file link --
curl can't click through it, and no stable direct URL is published.

Manual steps:
  1. Open $MARC_ITEM_URL in a browser.
  2. Download the ~12.8 GB zip (Books (All) MARC Records, 2019).
  3. Save it as: $MARC_ZIP
  4. Re-run this script (or 'scripts/download_data.sh marc') to extract it.

EOF
}

download_marc() {
    mkdir -p "$RAW_DIR"

    if compgen -G "$RAW_DIR"/*.xml.gz > /dev/null; then
        echo "MARC .xml.gz files already present in $RAW_DIR, skipping download."
        return 0
    fi

    if [ ! -f "$MARC_ZIP" ]; then
        echo "Attempting to download LOC 'Books (All)' 2019 MARC dataset (~12.8 GB)..."
        if ! curl -fL -A "$UA" -o "$MARC_ZIP" "$MARC_ITEM_URL" || ! file "$MARC_ZIP" | grep -qi zip; then
            rm -f "$MARC_ZIP"
            manual_marc_instructions
            return 1
        fi
    fi

    echo "Extracting $MARC_ZIP..."
    local extract_tmp
    extract_tmp="$(mktemp -d "$RAW_DIR/_extract_tmp.XXXXXX")"
    unzip -oq "$MARC_ZIP" -d "$extract_tmp"
    # The archive's internal layout isn't guaranteed, so pull every .xml.gz
    # out flat into data/raw/ regardless of nesting -- data_collection.py
    # just globs *.xml.gz directly inside --input_dir.
    find "$extract_tmp" -name '*.xml.gz' -exec mv {} "$RAW_DIR/" \;
    rm -rf "$extract_tmp"
    rm -f "$MARC_ZIP"

    if ! compgen -G "$RAW_DIR"/*.xml.gz > /dev/null; then
        echo "Extraction produced no .xml.gz files -- check $MARC_ZIP's contents by hand." >&2
        return 1
    fi
    echo "Done: $(ls "$RAW_DIR"/*.xml.gz | wc -l | tr -d ' ') .xml.gz files in $RAW_DIR"
}

download_geonames() {
    mkdir -p "$GEO_DIR"

    if [ ! -f "$GEO_DIR/admin1CodesASCII.txt" ]; then
        echo "Fetching admin1CodesASCII.txt..."
        curl -fsSL -o "$GEO_DIR/admin1CodesASCII.txt" \
            https://download.geonames.org/export/dump/admin1CodesASCII.txt
    fi

    # US CA GB AU: the only countries the direct geocoding pass needs.
    # The rest: residual countries the LLM geocoding pass's normalized
    # answers name but place_name_008 never does. See README's Step 3.
    local countries=(
        US CA GB AU
        DE IT FR IE IN JP ES CN CH RU MX NL SE PL FI RO BR IL DK CU ZA NG
        BE AR PR TW UA AT NZ LB KR CZ NO PT PH GR VI EG TR EC IR BA CL
    )

    for cc in "${countries[@]}"; do
        if [ -f "$GEO_DIR/${cc}.txt" ]; then
            continue
        fi
        echo "Fetching GeoNames ${cc}..."
        curl -fsSL -o "$GEO_DIR/${cc}.zip" "https://download.geonames.org/export/dump/${cc}.zip"
        unzip -oq "$GEO_DIR/${cc}.zip" "${cc}.txt" -d "$GEO_DIR"
        rm "$GEO_DIR/${cc}.zip"
    done
    echo "Done: GeoNames reference data in $GEO_DIR"
}

case "${1:-all}" in
    marc) download_marc ;;
    geonames) download_geonames ;;
    all)
        download_marc
        download_geonames
        ;;
    *)
        echo "Usage: $0 [marc|geonames]" >&2
        exit 1
        ;;
esac
