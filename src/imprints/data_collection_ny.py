"""Data collection for identifying New York publications.

Iterates through all MARCXML .xml.gz records, extracts LCCN, place of publication,
year of publication, and flags whether the place meets normalized criteria for New York.
"""

import os
import argparse
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from imprints.data_collection import get_xml_gzs, parse_records, extract_subfields, NUM_WORKERS
from imprints.data_cleaning import clean_string, get_target_cities, get_years_ints


def process_record_ny(record):
    """Extract rows for each place of publication with New York indicator."""
    ns = {"marc": "http://www.loc.gov/MARC21/slim"}
    lccn_vals = extract_subfields(record, "010", "a", ns)
    lccn = lccn_vals[0] if lccn_vals else None

    years_260 = extract_subfields(record, "260", "c", ns)
    years_264 = extract_subfields(record, "264", "c", ns)
    year_pub = get_years_ints(years_260 + years_264)

    places_260 = extract_subfields(record, "260", "a", ns)
    places_264 = extract_subfields(record, "264", "a", ns)
    places = list(dict.fromkeys(places_260 + places_264))
    if not places:
        places = [None]

    rows = []
    for place in places:
        place_clean = clean_string(place)
        city_group = get_target_cities(place_clean)
        is_new_york = city_group == "New York"
        rows.append({
            "lccn": lccn,
            "place_of_publication": place,
            "year_of_publication": year_pub,
            "is_new_york": is_new_york,
        })
    return rows


def process_file_ny(file_path):
    """Process a single .xml.gz file and return rows for NY data."""
    rows = []
    for record in parse_records(file_path):
        rows.extend(process_record_ny(record))
    return rows

def collect_ny(input_dir):
    """Iterate through all .xml.gz files in input_dir in parallel and collect data."""
    file_list = get_xml_gzs(input_dir)
    all_rows = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {executor.submit(process_file_ny, fp): fp for fp in file_list}
        for future in as_completed(future_to_file):
            fp = future_to_file[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                print(f"Processed {os.path.basename(fp)}: {len(rows)} rows")
            except Exception as e:
                print(f"Error processing {os.path.basename(fp)}: {e}")
    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Collect New York publication locations from MARCXML records."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory with input .xml.gz files"
    )
    parser.add_argument(
        "--output_csv", required=True, help="CSV file to write the output"
    )
    args = parser.parse_args()

    print(f"Scanning directory: {args.input_dir}")
    start = time.time()
    rows = collect_ny(args.input_dir)
    df = pd.DataFrame(
        rows, columns=["lccn", "place_of_publication", "year_of_publication", "is_new_york"]
    )
    df.to_csv(args.output_csv, index=False)
    elapsed = time.time() - start
    print(
        f"Processed {len(rows)} rows in {elapsed:.1f}s. Output written to {args.output_csv}"
    )


if __name__ == "__main__":
    main()