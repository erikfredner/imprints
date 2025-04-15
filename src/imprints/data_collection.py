from bs4 import BeautifulSoup
import gc
import gzip
import os
import pandas as pd
import pickle
import time
import xml.etree.ElementTree as ET


def get_xml_gzs(path):
    """Gets all of the xml.gz files in MDSConnect excluding the combined files, which duplicate records."""
    return [
        os.path.join(path, x)
        for x in os.listdir(path)
        if x.endswith(".xml.gz") and "combined" not in x
    ]


def open_gzip_file(file_path):
    """
    Function to open a gzipped file and return an iterator over its lines.
    """
    return gzip.open(file_path, "rt", encoding="utf-8")


def parse_records(file_path):
    """
    Generator function to parse records from a gzipped XML file.
    """
    with open_gzip_file(file_path) as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag == "{http://www.loc.gov/MARC21/slim}record":
                yield elem


def extract_subfields(record, tag, subfield_code, ns):
    """
    Extract subfield values for a given tag and subfield code.
    """
    return [
        subfield.text
        for subfield in record.findall(
            f'marc:datafield[@tag="{tag}"]/marc:subfield[@code="{subfield_code}"]', ns
        )
    ]


def process_record(record):
    """
    Function to process each <record> element.
    """
    ns = {"marc": "http://www.loc.gov/MARC21/slim"}
    classifications = extract_subfields(record, "050", "a", ns)
    if not classifications:
        return None

    # Only retain PS records
    if not any(
        str(classification).startswith("PS") for classification in classifications
    ):
        return None

    # Extract data from the <record> element
    lccn = extract_subfields(record, "010", "a", ns)
    personal_name_100 = extract_subfields(record, "100", "a", ns)
    title = extract_subfields(record, "245", "a", ns)
    years_260 = extract_subfields(record, "260", "c", ns)
    years_264 = extract_subfields(record, "264", "c", ns)
    places_260 = extract_subfields(record, "260", "a", ns)
    publisher_260 = extract_subfields(record, "260", "b", ns)
    places_264 = extract_subfields(record, "264", "a", ns)

    # Combine years and places
    years = list(set(years_260 + years_264))
    places = list(set(places_260 + places_264))

    data = {
        "lccn": lccn if lccn else None,
        "classifications": classifications,
        "title": title[0] if title else None,
        "year": years,
        "places": places,
        "publishers": publisher_260,
        "first_author": personal_name_100[0] if personal_name_100 else None,
    }
    return data


def process_files(file_paths, output_dir):
    """
    Function to process multiple gzipped XML files and save extracted data as pickles iteratively.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_files = len(file_paths)
    for i, file_path in enumerate(file_paths, 1):
        start_time = time.time()
        all_data = []
        for record in parse_records(file_path):
            data = process_record(record)
            if data:  # Only append data if it passed the filter
                all_data.append(data)

        # Generate a unique filename based on the input file name
        output_file = (
            f"{output_dir}/{os.path.basename(file_path).replace('.xml.gz', '')}.pkl"
        )

        # Write the collected data to a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(all_data, f)

        # Print progress
        print(
            f"Processed {i}/{total_files} files: {file_path} written to {output_file}"
        )
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

        del all_data
        gc.collect()


def load_pickles_to_dataframe(pickle_dir):
    """
    Load all pickle files from the specified directory into a single DataFrame.

    Parameters:
    pickle_dir (str): The directory containing the pickle files.

    Returns:
    pd.DataFrame: A DataFrame containing the concatenated data from all pickle files.
    """
    all_data = []

    for file_name in os.listdir(pickle_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(pickle_dir, file_name)
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                all_data.extend(data)  # Extend the list with data from each pickle file

    # Convert the combined data into a DataFrame
    df = pd.DataFrame(all_data)
    return df
