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
                # Free the record *and* drop already-processed siblings so the
                # root does not accumulate emptied <record> nodes across a
                # multi-GB file. Without this, memory grows unbounded.
                elem.clear()
                parent = elem.getparent()
                if parent is not None:
                    while elem.getprevious() is not None:
                        del parent[0]


def _local_name(tag):
    """Strip the namespace from a fully-qualified lxml tag."""
    return tag.rpartition("}")[2] if isinstance(tag, str) else tag


def collect_subfields(record):
    """Walk a record's datafields once and bucket the subfields we need.

    Returns a dict keyed by (tag, subfield_code) -> list of non-empty texts,
    plus the special key ("264", code, "pub") restricted to publication
    indicators. This single pass replaces ~10 namespaced findall() calls per
    record, which matters at dataset scale.

    For 264, the 2nd indicator distinguishes the function of the field
    (0=production, 1=publication, 2=distribution, 3=manufacture,
    4=copyright). We keep place ($a) and publisher ($b) only for publication
    (blank or "1"); $c dates are kept regardless because a copyright year is a
    valid publication-year proxy.
    """
    buckets = {}
    for field in record:
        if _local_name(field.tag) != "datafield":
            continue
        tag = field.get("tag")
        if tag not in ("010", "050", "090", "100", "245", "260", "264"):
            continue
        ind2 = field.get("ind2", " ")
        is_pub_264 = tag != "264" or ind2 in (" ", "1")
        for sub in field:
            if _local_name(sub.tag) != "subfield":
                continue
            text = sub.text
            if text is None or not text.strip():
                continue
            code = sub.get("code")
            # Restrict 264 place/publisher to publication-function fields.
            if tag == "264" and code in ("a", "b") and not is_pub_264:
                continue
            buckets.setdefault((tag, code), []).append(text)
    return buckets


def _dedup(*lists):
    """Order-preserving concat + dedup of several lists."""
    merged = []
    for lst in lists:
        merged.extend(lst)
    return list(dict.fromkeys(merged))


SUBJECT_GENRE_TAGS = ("600", "610", "611", "630", "650", "651", "655")


def collect_subject_occurrences(record):
    """Walk a record's datafields, capturing each 600/610/611/630/650/651/655
    occurrence as its own structured dict, in document order:
        {"tag": str, "ind2": str, "subfields": [(code, text), ...]}

    Separate from collect_subfields(), which flattens every occurrence of a
    tag into one list per (tag, code) -- fine for fields the rest of the
    pipeline only ever needs the first/merged value from, but it loses which
    subfields belonged to the same field instance. The secondary-literature
    classifier needs that grouping (e.g. to compare a single 600's $a/$d
    against the record's own 100, or to read a 650's $x/$v subdivision chain
    as written). Repeated codes within one occurrence (two $x in one 650) are
    kept in order, not deduped. All subfields present are captured, not
    pre-filtered to a fixed code list, so rule logic owns which codes it
    reads.
    """
    occurrences = []
    for field in record:
        if _local_name(field.tag) != "datafield":
            continue
        tag = field.get("tag")
        if tag not in SUBJECT_GENRE_TAGS:
            continue
        subfields = []
        for sub in field:
            if _local_name(sub.tag) != "subfield":
                continue
            text = sub.text
            if text is None or not text.strip():
                continue
            subfields.append((sub.get("code"), text))
        occurrences.append(
            {"tag": tag, "ind2": field.get("ind2", " "), "subfields": subfields}
        )
    return occurrences


def extract_008(record):
    """Return the raw 008 controlfield text, or None if absent/empty.

    Byte-position interpretation (literary form, biography code) is
    classification-specific and is owned by the consumer, not here.
    """
    for field in record:
        if _local_name(field.tag) != "controlfield":
            continue
        if field.get("tag") == "008":
            return field.text
    return None


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
    # A one-letter range is a top-level LC class and includes its subclasses
    # (e.g. P includes PR and PS).  Longer ranges identify a complete alpha
    # subclass, so PS must not also admit malformed/other classes such as PSA.
    prefix_matches = bool(cls) and (
        cls == prefix or (len(prefix) == 1 and cls.startswith(prefix))
    )
    if not prefix_matches:
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
    buckets = collect_subfields(record)
    classifications = buckets.get(("050", "a"), [])
    matches_class_range = filter_classification(classifications, class_range)

    lccn = buckets.get(("010", "a"), [])
    local_call_numbers = buckets.get(("090", "a"), [])
    personal_name_100 = buckets.get(("100", "a"), [])
    personal_name_100_dates = buckets.get(("100", "d"), [])
    title = buckets.get(("245", "a"), [])
    subtitle = buckets.get(("245", "b"), [])
    # 260 has no function indicator; its publication 264 counterparts were
    # filtered in collect_subfields. Merge order-preserving, deduped.
    years = _dedup(buckets.get(("260", "c"), []), buckets.get(("264", "c"), []))
    places = _dedup(buckets.get(("260", "a"), []), buckets.get(("264", "a"), []))
    publishers = _dedup(buckets.get(("260", "b"), []), buckets.get(("264", "b"), []))

    data = {
        "lccn": lccn[0] if lccn else None,
        "classifications": classifications,
        "matches_class_range": matches_class_range,
        "title": title[0] if title else None,
        "subtitle": subtitle[0] if subtitle else None,
        "year": years,
        "places": places,
        "publishers": publishers,
        "first_author": personal_name_100[0] if personal_name_100 else None,
        # Fields below support imprints.secondary_classification and are
        # otherwise unused by the rest of the pipeline.
        "local_call_numbers": local_call_numbers,
        "first_author_dates": personal_name_100_dates[0]
        if personal_name_100_dates
        else None,
        "subject_genre_fields": collect_subject_occurrences(record),
        "field_008": extract_008(record),
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
            all_data.append(data)
        with open(output_file, "wb") as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return f"Processed: {os.path.basename(file_path)} ({len(all_data)} records, {time.time() - start_time:.1f}s)"
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
        description="Extract MARC xml.gz files and mark records that match an LC range."
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

    # Step 2: Process files in parallel (all records kept, matching flagged)
    process_files_parallel(file_list, output_dir, class_range)
