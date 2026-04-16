"""PubMed E-utilities client for metadata collection.

Uses Bio.Entrez for PubMed/PMC queries. Respects NCBI rate limits:
- 3 requests/sec without API key
- 10 requests/sec with NCBI_API_KEY

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

import os
import time
import xml.etree.ElementTree as ET
from typing import Optional

from Bio import Entrez

from src.utils.config import load_config

# Entrez requires an email
Entrez.email = "synthsearch@research.example.com"
_api_key = os.environ.get("NCBI_API_KEY")
if _api_key:
    Entrez.api_key = _api_key

# Rate limiting
_RATE_LIMIT = 10 if _api_key else 3  # requests per second
_MIN_INTERVAL = 1.0 / _RATE_LIMIT
_last_request_time = 0.0


def _rate_limit():
    """Enforce NCBI rate limiting."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def search_pubmed(
    query: str,
    max_results: int = 10000,
    min_date: str = "2015/01/01",
    max_date: str = "2026/12/31",
    sort: str = "relevance",
) -> list[str]:
    """Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed search query (supports Boolean and MeSH).
        max_results: Maximum number of results to return.
        min_date: Minimum publication date (YYYY/MM/DD).
        max_date: Maximum publication date (YYYY/MM/DD).
        sort: Sort order ('relevance' or 'pub_date').

    Returns:
        List of PMID strings.
    """
    _rate_limit()
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        mindate=min_date,
        maxdate=max_date,
        sort=sort,
        usehistory="y",
    )
    results = Entrez.read(handle)
    handle.close()

    count = int(results["Count"])
    pmids = list(results["IdList"])

    # If more results than retmax, fetch in batches using history
    if count > max_results:
        print(f"  PubMed returned {count} total results, capped at {max_results}")

    if count > len(pmids) and count <= max_results:
        webenv = results["WebEnv"]
        query_key = results["QueryKey"]
        batch_size = 10000
        for start in range(len(pmids), min(count, max_results), batch_size):
            _rate_limit()
            h = Entrez.esearch(
                db="pubmed",
                term=query,
                retstart=start,
                retmax=batch_size,
                mindate=min_date,
                maxdate=max_date,
                sort=sort,
                WebEnv=webenv,
                query_key=query_key,
                usehistory="y",
            )
            batch = Entrez.read(h)
            h.close()
            pmids.extend(batch["IdList"])

    return pmids[:max_results]


def fetch_pubmed_records(
    pmids: list[str],
    batch_size: int = 200,
) -> list[dict]:
    """Fetch full metadata records for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs.
        batch_size: Number of records per API call (max 10000).

    Returns:
        List of parsed record dictionaries.
    """
    records = []
    total = len(pmids)

    for i in range(0, total, batch_size):
        batch = pmids[i : i + batch_size]
        _rate_limit()

        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="xml",
            retmode="xml",
        )
        xml_data = handle.read()
        handle.close()

        parsed = _parse_pubmed_xml(xml_data)
        records.extend(parsed)

        if (i + batch_size) % 1000 == 0 or i + batch_size >= total:
            print(f"  Fetched {min(i + batch_size, total)}/{total} PubMed records")

    return records


def _parse_pubmed_xml(xml_bytes: bytes) -> list[dict]:
    """Parse PubMed XML into structured records."""
    records = []
    root = ET.fromstring(xml_bytes)

    for article in root.findall(".//PubmedArticle"):
        try:
            record = _parse_single_article(article)
            if record:
                records.append(record)
        except Exception as e:
            pmid = article.findtext(".//PMID", default="unknown")
            print(f"  Warning: failed to parse PMID {pmid}: {e}")
            continue

    return records


def _parse_single_article(article: ET.Element) -> Optional[dict]:
    """Parse a single PubmedArticle XML element."""
    medline = article.find(".//MedlineCitation")
    if medline is None:
        return None

    pmid = medline.findtext("PMID", default="")
    if not pmid:
        return None

    art = medline.find("Article")
    if art is None:
        return None

    # Title
    title = art.findtext("ArticleTitle", default="")

    # Abstract
    abstract_parts = []
    abstract_el = art.find("Abstract")
    if abstract_el is not None:
        for text_el in abstract_el.findall("AbstractText"):
            label = text_el.get("Label", "")
            text = "".join(text_el.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # Year
    year = None
    pub_date = art.find(".//PubDate")
    if pub_date is not None:
        year_el = pub_date.findtext("Year")
        if year_el:
            year = int(year_el)
        else:
            medline_date = pub_date.findtext("MedlineDate", default="")
            if medline_date and len(medline_date) >= 4:
                try:
                    year = int(medline_date[:4])
                except ValueError:
                    pass

    # Journal
    journal = art.findtext(".//Journal/Title", default="")
    journal_abbrev = art.findtext(".//Journal/ISOAbbreviation", default="")

    # DOI
    doi = ""
    for eid in art.findall(".//ELocationID"):
        if eid.get("EIdType") == "doi":
            doi = eid.text or ""
            break
    if not doi:
        article_data = article.find(".//PubmedData")
        if article_data is not None:
            for aid in article_data.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text or ""
                    break

    # PMC ID
    pmcid = ""
    article_data = article.find(".//PubmedData")
    if article_data is not None:
        for aid in article_data.findall(".//ArticleId"):
            if aid.get("IdType") == "pmc":
                pmcid = aid.text or ""
                break

    # Authors
    authors = []
    author_list = art.find("AuthorList")
    if author_list is not None:
        for author in author_list.findall("Author"):
            last = author.findtext("LastName", default="")
            first = author.findtext("ForeName", default="")
            if last:
                authors.append(f"{last}, {first}" if first else last)

    # MeSH terms
    mesh_terms = []
    mesh_list = medline.find("MeshHeadingList")
    if mesh_list is not None:
        for mesh in mesh_list.findall("MeshHeading"):
            descriptor = mesh.findtext("DescriptorName", default="")
            if descriptor:
                mesh_terms.append(descriptor)
            for qualifier in mesh.findall("QualifierName"):
                if qualifier.text:
                    mesh_terms.append(f"{descriptor}/{qualifier.text}")

    # Publication types
    pub_types = []
    for pt in art.findall(".//PublicationTypeList/PublicationType"):
        if pt.text:
            pub_types.append(pt.text)

    # Keywords
    keywords = []
    for kw_list in medline.findall("KeywordList"):
        for kw in kw_list.findall("Keyword"):
            if kw.text:
                keywords.append(kw.text)

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": journal,
        "venue_abbrev": journal_abbrev,
        "authors": authors,
        "mesh_terms": mesh_terms,
        "publication_types": pub_types,
        "keywords": keywords,
        "source": "pubmed",
    }


def get_pmc_ids_for_pmids(pmids: list[str], batch_size: int = 200) -> dict[str, str]:
    """Convert PMIDs to PMCIDs using the ID converter.

    Args:
        pmids: List of PubMed IDs.
        batch_size: Batch size for API calls.

    Returns:
        Dict mapping PMID -> PMCID (only for papers that have a PMC ID).
    """
    pmid_to_pmc = {}

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        _rate_limit()

        handle = Entrez.elink(
            dbfrom="pubmed",
            db="pmc",
            id=batch,
            linkname="pubmed_pmc",
        )
        results = Entrez.read(handle)
        handle.close()

        for linked in results:
            from_id = linked["IdList"][0] if linked["IdList"] else None
            if from_id and linked.get("LinkSetDb"):
                for linkset in linked["LinkSetDb"]:
                    for link in linkset.get("Link", []):
                        pmc_id = link.get("Id", "")
                        if pmc_id:
                            pmid_to_pmc[from_id] = f"PMC{pmc_id}"
                            break

    return pmid_to_pmc
