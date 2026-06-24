"""Shared fixtures/helpers for building synthetic MARC records.

These mirror the real LoC "Books (All)" structure closely enough to exercise
extraction: the MARC21 slim namespace, datafields with indicators, and
repeated/empty subfields.
"""

import gzip

import pytest
from lxml import etree as ET

NS = "http://www.loc.gov/MARC21/slim"


def _subfield_xml(code, text):
    if text is None:
        return f'<subfield code="{code}"/>'
    return f'<subfield code="{code}">{text}</subfield>'


def build_record_xml(datafields):
    """Build a <record> XML string in the MARC slim namespace.

    `datafields` is a list of (tag, ind2, subfields) where subfields is a list
    of (code, text) pairs. A text of None produces an empty <subfield/>.
    """
    parts = [f'<record xmlns="{NS}">']
    for tag, ind2, subfields in datafields:
        parts.append(f'<datafield tag="{tag}" ind1=" " ind2="{ind2}">')
        parts.extend(_subfield_xml(code, text) for code, text in subfields)
        parts.append("</datafield>")
    parts.append("</record>")
    return "".join(parts)


def make_record(datafields):
    """Parse `build_record_xml` into an lxml element for process_record()."""
    return ET.fromstring(build_record_xml(datafields))


@pytest.fixture
def make_record_factory():
    return make_record


def write_marc_gz(path, records):
    """Write a gzipped MARC collection of the given record element-lists."""
    body = "".join(build_record_xml(df) for df in records)
    xml = (f'<collection xmlns="{NS}">{body}</collection>').encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(xml)
    return str(path)


@pytest.fixture
def marc_gz_writer():
    return write_marc_gz
