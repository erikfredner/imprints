import os
import gzip
import pickle
import time
import re
import pandas as pd
from lxml import etree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_WORKERS = os.cpu_count() or 4


def get_xml_gzs(path):
    return [
        os.path.join(path, x)
        for x in os.listdir(path)
        if x.endswith(".xml.gz") and "combined" not in x
    ]


def open_gzip_file(file_path):
    return gzip.open(file_path, "rb")


def parse_records(file_path):
    NS_MARC = "{http://www.loc.gov/MARC21/slim}"
    with open_gzip_file(file_path) as f:
        context = ET.iterparse(f, events=("end",), recover=True)
        for event, elem in context:
            if elem.tag == f"{NS_MARC}record":
                yield elem
                elem.clear()


def extract_subfields(record, tag, subfield_code, ns):
    return [
        subfield.text
        for subfield in record.findall(
            f'marc:datafield[@tag="{tag}"]/marc:subfield[@code="{subfield_code}"]', ns
        )
        if subfield.text is not None
    ]


def parse_class(class_str):
    """
    Split into prefix and digits.
    E.g. PS3555.123 -> ('PS', 3555)
    """
    m = re.match(r"([A-Z]+)(\d+)?", class_str)
    if not m:
        return None, None
    return m.group(1), int(m.group(2)) if m.group(2) else None


def matches_range(classification, prefix, num_min=None, num_max=None):
    """
    Test if a classification matches a given prefix and optional num range.
    E.g. classification 'PR6053' matches prefix 'PR', min 6050, max 6060.
    """
    # Can be list; scan all entries
    if isinstance(classification, (list, tuple)):
        return any(matches_range(c, prefix, num_min, num_max) for c in classification)
    if not classification or not isinstance(classification, str):
        return False
    cls, num = parse_class(classification.strip())
    if not cls or not classification.strip().startswith(prefix):
        return False
    if num_min is not None and (num is None or num < num_min):
        return False
    if num_max is not None and (num is None or num > num_max):
        return False
    return True


def filter_classification(classifications, class_range):
    """
    True iff *any* classification matches the class_range specifier.
    class_range: {'prefix':'PS', 'min':None, 'max':None} or include min/max for numeric range.
    """
    prefix = class_range["prefix"]
    minval = class_range.get("min")
    maxval = class_range.get("max")
    return matches_range(classifications, prefix, minval, maxval)


def parse_range_spec(range_str):
    """
    Examples:
        'PS' -> {'prefix':'PS'}
        'PR9000-PR9999' -> {'prefix':'PR', 'min':9000, 'max':9999}
        'PG' -> {'prefix':'PG'}
    Only supports one contiguous range or single prefix.
    """
    m = re.match(r"^([A-Z]+)(\d{0,})(-([A-Z]+)?(\d{1,}))?$", range_str)
    if not m:
        raise ValueError(f"Range spec not recognized: {range_str}")
    prefix = m.group(1)
    minval = int(m.group(2)) if m.group(2) else None
    maxval = int(m.group(5)) if m.group(5) else None
    return {"prefix": prefix, "min": minval, "max": maxval}


def process_record(record, class_range):
    ns = {"marc": "http://www.loc.gov/MARC21/slim"}
    classifications = extract_subfields(record, "050", "a", ns)
    if not classifications:
        return None
    if not filter_classification(classifications, class_range):
        return None

    lccn = extract_subfields(record, "010", "a", ns)
    personal_name_100 = extract_subfields(record, "100", "a", ns)
    title = extract_subfields(record, "245", "a", ns)
    years_260 = extract_subfields(record, "260", "c", ns)
    years_264 = extract_subfields(record, "264", "c", ns)
    places_260 = extract_subfields(record, "260", "a", ns)
    publisher_260 = extract_subfields(record, "260", "b", ns)
    places_264 = extract_subfields(record, "264", "a", ns)
    years = list(dict.fromkeys(years_260 + years_264))
    places = list(dict.fromkeys(places_260 + places_264))

    data = {
        "lccn": lccn[0] if lccn else None,
        "classifications": classifications,
        "title": title[0] if title else None,
        "year": years,
        "places": places,
        "publishers": publisher_260,
        "first_author": personal_name_100[0] if personal_name_100 else None,
    }
    return data


def process_single_file(args):
    file_path, output_dir, class_range = args
    out_basename = os.path.basename(file_path).replace(".xml.gz", ".pkl")
    output_file = os.path.join(output_dir, out_basename)
    all_data = []
    start_time = time.time()
    try:
        for record in parse_records(file_path):
            data = process_record(record, class_range)
            if data:
                all_data.append(data)
        with open(output_file, "wb") as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return f"Processed: {os.path.basename(file_path)} ({len(all_data)} records, {time.time()-start_time:.1f}s)"
    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {e}"


def process_files_parallel(file_paths, output_dir, class_range):
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_single_file, (fp, output_dir, class_range)): fp
            for fp in file_paths
        }
        for i, future in enumerate(as_completed(future_to_file), 1):
            result = future.result()
            print(f"[{i}/{len(file_paths)}] {result}")


def load_pickles_to_dataframe(pickle_dir):
    all_data = []
    for file_name in os.listdir(pickle_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(pickle_dir, file_name)
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                all_data.extend(data)
    df = pd.DataFrame(all_data)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract/filter MARC xml.gz files for arbitrary LC range."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with input .xml.gz files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output pickles"
    )
    parser.add_argument(
        "--class_range",
        type=str,
        required=True,
        help="LC class to extract: e.g. PS or PR9000-PR9999",
    )
    args = parser.parse_args()

    # Parse range, e.g. 'PR' or 'PR9000-PR9999'
    class_range = parse_range_spec(args.class_range)
    range_dir_name = (args.class_range).replace(":", "-").replace("/", "_")
    output_dir = os.path.join(args.output_dir, range_dir_name)

    # Step 1: Gather all matching input files
    file_list = get_xml_gzs(args.input_dir)
    print(f"Found {len(file_list)} files to process.")

    # Step 2: Process files in parallel for target LC range
    process_files_parallel(file_list, output_dir, class_range)

    # Optionally: Combine into DataFrame after processing (if needed)
    # df = load_pickles_to_dataframe(output_dir)
    # df.to_csv(f"all_{range_dir_name}.csv", index=False)
