"""OpenAlex API client for scholarly metadata enrichment.

OpenAlex provides citation graphs, concepts, topics, and institutional data.
API: https://docs.openalex.org/

Rate limits:
- Without polite pool: 10 req/sec
- With polite pool (email in User-Agent): 100K req/day, ~100 req/sec
"""

import time
from typing import Any, Optional

import requests

# Polite pool: set mailto for higher rate limits
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "SynthSearch/0.1 (mailto:synthsearch@research.example.com)",
    "Accept": "application/json",
})

_BASE_URL = "https://api.openalex.org"
_RATE_LIMIT = 10  # requests per second (conservative)
_MIN_INTERVAL = 1.0 / _RATE_LIMIT
_last_request_time = 0.0


def _rate_limit():
    """Enforce rate limiting."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict:
    """Make a GET request to OpenAlex API."""
    _rate_limit()
    url = f"{_BASE_URL}{endpoint}"
    resp = _SESSION.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_works(
    query: str | None = None,
    filter_str: str | None = None,
    per_page: int = 200,
    max_results: int = 10000,
    cursor: str = "*",
    select: str | None = None,
) -> list[dict]:
    """Search OpenAlex works (papers) with cursor pagination.

    Args:
        query: Full-text search query.
        filter_str: OpenAlex filter string, e.g.:
            'type:article,from_publication_date:2015-01-01,
             to_publication_date:2026-12-31,language:en'
        per_page: Results per page (max 200).
        max_results: Maximum total results.
        cursor: Pagination cursor (start with '*').
        select: Comma-separated field list to return.

    Returns:
        List of work dictionaries.
    """
    results = []
    params = {"per_page": min(per_page, 200)}

    if query:
        params["search"] = query
    if filter_str:
        params["filter"] = filter_str
    if select:
        params["select"] = select

    current_cursor = cursor
    while len(results) < max_results and current_cursor:
        params["cursor"] = current_cursor
        data = _get("/works", params)

        works = data.get("results", [])
        if not works:
            break

        results.extend(works)
        current_cursor = data.get("meta", {}).get("next_cursor")

        if len(results) % 1000 == 0 or not current_cursor:
            total_avail = data.get("meta", {}).get("count", "?")
            print(f"  OpenAlex: fetched {len(results)} / {total_avail}")

    return results[:max_results]


def get_work_by_doi(doi: str) -> Optional[dict]:
    """Fetch a single work by DOI."""
    try:
        return _get(f"/works/doi:{doi}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def get_work_by_pmid(pmid: str) -> Optional[dict]:
    """Fetch a single work by PubMed ID."""
    try:
        return _get(f"/works/pmid:{pmid}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def get_works_by_pmids(
    pmids: list[str],
    batch_size: int = 50,
    select: str | None = None,
) -> dict[str, dict]:
    """Fetch multiple works by PMID using filter.

    Args:
        pmids: List of PubMed IDs.
        batch_size: PMIDs per request (max ~50 for URL length).
        select: Fields to return.

    Returns:
        Dict mapping PMID -> work record.
    """
    result = {}

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        # OpenAlex filter: pmid:<id1>|<id2>|...
        pmid_filter = "|".join(batch)
        filter_str = f"ids.pmid:{pmid_filter}"

        params = {"filter": filter_str, "per_page": batch_size}
        if select:
            params["select"] = select

        data = _get("/works", params)
        for work in data.get("results", []):
            # Extract PMID from work IDs
            work_pmid = _extract_pmid(work)
            if work_pmid:
                result[work_pmid] = work

        if (i + batch_size) % 500 == 0:
            print(f"  OpenAlex enrichment: {min(i + batch_size, len(pmids))}/{len(pmids)}")

    return result


def parse_work(work: dict) -> dict:
    """Parse an OpenAlex work into our corpus schema.

    Args:
        work: Raw OpenAlex work dictionary.

    Returns:
        Parsed record with fields matching our corpus schema.
    """
    # IDs
    openalex_id = work.get("id", "")
    doi = (work.get("doi") or "").replace("https://doi.org/", "")

    pmid = _extract_pmid(work)
    pmcid = ""
    for loc in work.get("locations", []):
        source = loc.get("source") or {}
        if source.get("type") == "repository" and "ncbi.nlm.nih.gov/pmc" in (loc.get("landing_page_url") or ""):
            # Extract PMC ID from URL
            url = loc.get("landing_page_url", "")
            if "PMC" in url:
                parts = url.split("/")
                for p in parts:
                    if p.startswith("PMC"):
                        pmcid = p.rstrip("/")
                        break

    # Basics
    title = work.get("title") or ""
    year = work.get("publication_year")
    cited_by_count = work.get("cited_by_count", 0)

    # Venue
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name", "")

    # Abstract (OpenAlex stores inverted index)
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # Authors
    authors = []
    for authorship in work.get("authorships", []):
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    # Concepts / topics
    concepts = []
    for concept in work.get("concepts", []):
        if concept.get("score", 0) > 0.3:  # filter low-confidence
            concepts.append({
                "name": concept.get("display_name", ""),
                "score": concept.get("score", 0),
                "level": concept.get("level", 0),
            })

    topics = []
    for topic in work.get("topics", []):
        topics.append({
            "name": topic.get("display_name", ""),
            "subfield": topic.get("subfield", {}).get("display_name", ""),
            "field": topic.get("field", {}).get("display_name", ""),
        })

    # Citation links
    referenced_works = [
        w.replace("https://openalex.org/", "")
        for w in work.get("referenced_works", [])
    ]
    # cited_by requires a separate query (too expensive in bulk)

    # Open access status
    oa = work.get("open_access") or {}
    is_oa = oa.get("is_oa", False)
    oa_url = oa.get("oa_url", "")

    return {
        "openalex_id": openalex_id,
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": venue,
        "authors": authors,
        "concepts": concepts,
        "topics": topics,
        "cited_by_count": cited_by_count,
        "referenced_work_ids": referenced_works,
        "is_oa": is_oa,
        "oa_url": oa_url,
        "source": "openalex",
    }


def _extract_pmid(work: dict) -> str:
    """Extract PubMed ID from OpenAlex work."""
    ids = work.get("ids", {})
    pmid = ids.get("pmid", "")
    if pmid:
        # Format: https://pubmed.ncbi.nlm.nih.gov/12345678
        return pmid.split("/")[-1]
    return ""


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""

    # inverted_index: {"word": [position1, position2, ...], ...}
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)
