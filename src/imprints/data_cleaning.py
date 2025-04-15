from bs4 import BeautifulSoup
from thefuzz import fuzz
import gc
import gzip
import io
import numpy as np
import os
import pandas as pd
import pickle
import re
import string
import sys
import time
import xml.etree.ElementTree as ET


def get_ps_digits(classification):
    """
    Extract the digits following the PS prefix in the classification.
    """
    match = re.search(r"PS(.*)", classification)
    if match:
        s = match.group(1).strip()
    # drop any remaining letters; only retain numbers
    try:
        if "." in s:
            return int(s.split(".")[0].strip())
        elif " " in s:
            return int(s.split(" ")[0].strip())
        else:
            return int(classification.split("PS")[-1].split(".")[0].strip())
    except:
        return None


def get_ps_digits(classification):
    """
    Extract digits immediately following "PS" up to but not including the first non-numeric character,
    or until the end of the string if no non-numeric character follows.
    """
    match = re.search(r"PS(\d+)(?=[^\d]|$)", classification)
    if match:
        return int(match.group(1))
    else:
        return None


def get_year_int(year):
    """
    Function to extract the year as an integer from the year column.
    """
    match = re.search(r"\d{4}", year)
    return int(match.group()) if match else None


def get_years_ints(year):
    if type(year) is str:
        return get_year_int(year)
    # Resolve cases with multiple years to the earliest year:
    elif type(year) is list:
        y_temp = [get_year_int(y) for y in year]
        y_temp = [y for y in y_temp if y is not None]
        if y_temp:
            return min(y_temp)
        else:
            return None


def get_publishers_year_ints(publishers):
    """Get year-like values from publishers, and take the minimum value.
    (Rarely, LC records include publication year in the publisher field.)"""
    if type(publishers) is str:
        publishers = [publishers]
    try:
        return min([get_year_int(p) for p in publishers if get_year_int(p) is not None])
    except ValueError:
        return None


def get_min_year(row):
    return min(row["year_publisher_int"], row["publisher_year_int"])


def clean_col(df, col):
    return df[col].apply(lambda x: x[0] if len(x) == 1 else x)


# clean up place names
def clean_string(s):

    if s is None:
        return None

    # Remove all characters other than A-Z (upper or lower) or space
    s = re.sub(r"[^a-zA-Z\s]", "", s)

    # Lowercase everything
    s = s.lower()

    # Strip whitespace
    s = s.strip()

    return s


def get_decade(year):
    if year > 0:
        return year // 10 * 10


def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def get_target_cities(
    placename, target_cities=["Boston", "Philadelphia", "New York"], threshold=85
):
    """
    Identifies occurrences of target cities in a messy placename string using an edit distance metric.
    """
    if pd.isna(placename) or not placename.strip():
        return "No place of publication"

    cleaned_cities = list()
    for target_city in target_cities:
        if fuzz.partial_ratio(target_city.lower(), placename.lower()) >= threshold:
            cleaned_cities.append(target_city)

    if cleaned_cities:
        return cleaned_cities
    else:
        return "Other"
