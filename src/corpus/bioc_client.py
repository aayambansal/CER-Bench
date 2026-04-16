"""PMC BioC API client for full-text article retrieval.

BioC provides PMC Open Access articles in structured XML/JSON format,
with sections, paragraphs, and annotations pre-parsed.

API: https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
"""

import json
import time
from typing import Optional

import requests

_SESSION = requests.Session()
_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
_RATE_LIMIT = 3  # requests per second
_MIN_INTERVAL = 1.0 / _RATE_LIMIT
_last_request = 0.0


def _rate_limit():
    global _last_request
    now = time.time()
    elapsed = now - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request = time.time()


def fetch_bioc_article(pmcid: str) -> Optional[dict]:
    """Fetch a PMC article in BioC JSON format.

    Args:
        pmcid: PMC ID (e.g., 'PMC7096066'). Strips 'PMC' prefix if present.

    Returns:
        BioC JSON document dict, or None if not available.
    """
    clean_id = pmcid.replace("PMC", "")
    url = f"{_BASE_URL}/{clean_id}/unicode"
    _rate_limit()

    try:
        resp = _SESSION.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            resp.raise_for_status()
    except Exception as e:
        print(f"  Warning: BioC fetch failed for {pmcid}: {e}")
        return None


def parse_bioc_to_sections(bioc_doc: dict) -> dict:
    """Parse BioC JSON into our structured document format.

    Returns:
        Dict with title, abstract, sections, figure_captions, table_texts.
    """
    if not bioc_doc or "documents" not in bioc_doc:
        return {}

    doc = bioc_doc["documents"][0] if bioc_doc.get("documents") else {}
    passages = doc.get("passages", [])

    title = ""
    abstract = ""
    sections = []
    figure_captions = []
    table_texts = []

    for passage in passages:
        infons = passage.get("infons", {})
        section_type = infons.get("section_type", "").lower()
        p_type = infons.get("type", "").lower()
        text = passage.get("text", "")

        if not text or len(text.strip()) < 10:
            continue

        if p_type == "front" or section_type == "title":
            if not title:
                title = text.strip()

        elif p_type == "abstract" or section_type == "abstract":
            abstract += " " + text.strip()

        elif "fig" in section_type or "figure" in p_type:
            figure_captions.append({
                "figure_id": infons.get("id", ""),
                "label": infons.get("label", ""),
                "caption": text.strip()[:1000],
            })

        elif "table" in section_type or "table" in p_type:
            table_texts.append({
                "table_id": infons.get("id", ""),
                "label": infons.get("label", ""),
                "caption": text.strip()[:1000],
                "body_text": "",
            })

        else:
            # Regular section paragraph
            heading = infons.get("section", "")
            # Classify section type
            st = "other"
            heading_lower = heading.lower()
            if any(k in heading_lower for k in ["introduction", "background"]):
                st = "introduction"
            elif any(k in heading_lower for k in ["method", "material", "procedure"]):
                st = "methods"
            elif any(k in heading_lower for k in ["result", "finding"]):
                st = "results"
            elif any(k in heading_lower for k in ["discussion", "interpretation"]):
                st = "discussion"
            elif any(k in heading_lower for k in ["conclusion", "summary"]):
                st = "conclusion"

            sections.append({
                "heading": heading,
                "section_type": st,
                "parent_heading": "",
                "text": text.strip(),
                "char_count": len(text),
            })

    return {
        "title": title.strip(),
        "abstract": abstract.strip(),
        "sections": sections,
        "figure_captions": figure_captions,
        "table_texts": table_texts,
        "section_count": len(sections),
        "figure_count": len(figure_captions),
        "table_count": len(table_texts),
        "has_methods": any(s["section_type"] == "methods" for s in sections),
        "has_results": any(s["section_type"] == "results" for s in sections),
        "total_chars": sum(s["char_count"] for s in sections),
    }
