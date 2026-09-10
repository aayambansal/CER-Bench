"""Legacy optional Bio.Entrez transport; offline parsing is stdlib-only.

The authoritative rebuild does NOT call this network transport: source XML is
retrieved exclusively through WebFetch. Importing this module reads no secrets
or configuration and does not import Bio/yaml. Anonymous limit: 3 requests/sec.
"""

import time
import xml.etree.ElementTree as ET
from src.corpus.pubmed_xml import parse_records as _parse_pubmed_xml
from src.corpus.pubmed_xml import parse_article, local_tree

_last_request_time = 0.0


def _entrez():
    from Bio import Entrez
    Entrez.email = "synthsearch@research.example.com"
    return Entrez


def _rate_limit():
    global _last_request_time
    time.sleep(max(0, 1 / 3 - (time.monotonic() - _last_request_time)))
    _last_request_time = time.monotonic()


def _parse_single_article(article):
    record, issues = parse_article(local_tree(ET.tostring(article)))
    if issues:
        raise ValueError("Invalid PubMed article: " + ", ".join(issues))
    return record


def search_pubmed(query, max_results=10000, min_date="2015/01/01",
                  max_date="2026/12/31", sort="relevance"):
    entrez = _entrez()
    pmids = []
    while len(pmids) < max_results:
        _rate_limit()
        with entrez.esearch(db="pubmed", term=query, retstart=len(pmids),
                            retmax=min(10000, max_results - len(pmids)),
                            mindate=min_date, maxdate=max_date, sort=sort) as handle:
            result = entrez.read(handle)
        batch = list(result["IdList"])
        pmids.extend(batch)
        if not batch or len(pmids) >= int(result["Count"]):
            break
    return pmids[:max_results]


def fetch_pubmed_records(pmids, batch_size=200):
    if not 1 <= batch_size <= 200:
        raise ValueError("batch_size must be 1..200")
    entrez = _entrez()
    records = []
    for i in range(0, len(pmids), batch_size):
        _rate_limit()
        with entrez.efetch(db="pubmed", id=",".join(pmids[i:i + batch_size]),
                           rettype="xml", retmode="xml") as handle:
            records.extend(_parse_pubmed_xml(handle.read()))
    return records


def get_pmc_ids_for_pmids(pmids, batch_size=200):
    # Do not infer own identity from references or an unconstrained link graph.
    return {r["pmid"]: r["pmcid"] for r in fetch_pubmed_records(pmids, batch_size) if r["pmcid"]}
