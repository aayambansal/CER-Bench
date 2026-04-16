#!/usr/bin/env python3
"""Script 03: Parse PMC XML into structured documents.

Parses PMC full-text XML and extracts:
- Title, abstract (from metadata if needed)
- Sections with headings and section types
- Figure captions
- Table text (captions + body)

Outputs parsed documents to data/interim/parsed/

Usage:
    python scripts/03_parse_documents.py [--max N]
"""

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir


# ─── Section type classification ─────────────────────────────────────────────

_SECTION_PATTERNS = {
    "introduction": re.compile(
        r"^(introduction|background|overview)$", re.IGNORECASE
    ),
    "methods": re.compile(
        r"^(methods?|materials?\s*(and|&)\s*methods?|experimental|"
        r"study\s*design|procedures?|methodology)$",
        re.IGNORECASE,
    ),
    "results": re.compile(
        r"^(results?|findings|outcomes?)$", re.IGNORECASE
    ),
    "discussion": re.compile(
        r"^(discussion|interpretation)$", re.IGNORECASE
    ),
    "conclusion": re.compile(
        r"^(conclusions?|summary|concluding\s*remarks?)$", re.IGNORECASE
    ),
    "results_discussion": re.compile(
        r"^(results?\s*(and|&)\s*discussion)$", re.IGNORECASE
    ),
}


def classify_section(heading: str) -> str:
    """Classify a section heading into a standard type."""
    heading_clean = heading.strip()
    for section_type, pattern in _SECTION_PATTERNS.items():
        if pattern.match(heading_clean):
            return section_type
    return "other"


# ─── XML parsing ─────────────────────────────────────────────────────────────

def _get_text(element: ET.Element) -> str:
    """Recursively extract all text from an XML element, including tails."""
    if element is None:
        return ""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        # Skip certain elements
        tag = child.tag
        if tag in ("xref", "ext-link", "uri"):
            # Keep the text content of references
            if child.text:
                parts.append(child.text)
        elif tag == "sup":
            if child.text:
                parts.append(child.text)
        elif tag == "sub":
            if child.text:
                parts.append(child.text)
        elif tag in ("bold", "italic", "underline", "monospace"):
            parts.append(_get_text(child))
        elif tag == "table-wrap" or tag == "fig":
            pass  # Skip inline figures/tables
        elif tag == "disp-formula" or tag == "inline-formula":
            parts.append("[formula]")
        else:
            parts.append(_get_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove excessive punctuation artifacts
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text


def parse_pmc_xml(xml_path: Path) -> Optional[dict]:
    """Parse a single PMC XML file into a structured document.

    Returns:
        Dict with parsed document fields, or None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Warning: XML parse error in {xml_path.name}: {e}")
        return None

    # Find the article element
    article = root.find(".//article")
    if article is None:
        article = root  # Sometimes the root IS the article

    # Front matter
    front = article.find(".//front")
    body = article.find(".//body")

    # PMC ID from filename
    pmcid = xml_path.stem

    # Title
    title = ""
    if front is not None:
        title_el = front.find(".//article-title")
        if title_el is not None:
            title = _get_text(title_el)

    # Abstract
    abstract = ""
    if front is not None:
        abstract_el = front.find(".//abstract")
        if abstract_el is not None:
            abstract_parts = []
            for child in abstract_el:
                if child.tag == "sec":
                    # Structured abstract with sections
                    sec_title = child.findtext("title", default="")
                    sec_text = _get_text(child.find("p")) if child.find("p") is not None else _get_text(child)
                    if sec_title:
                        abstract_parts.append(f"{sec_title}: {sec_text}")
                    else:
                        abstract_parts.append(sec_text)
                elif child.tag == "p":
                    abstract_parts.append(_get_text(child))
                elif child.tag == "title":
                    pass  # skip abstract title
                else:
                    text = _get_text(child)
                    if text:
                        abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
            if not abstract:
                abstract = _get_text(abstract_el)

    # Body sections
    sections = []
    if body is not None:
        for sec in body.findall(".//sec"):
            # Get heading
            title_el = sec.find("title")
            heading = _get_text(title_el) if title_el is not None else ""

            # Get paragraph text (only direct child <p> elements)
            paragraphs = []
            for p in sec.findall("p"):
                p_text = _clean_text(_get_text(p))
                if p_text and len(p_text) > 20:  # Skip tiny fragments
                    paragraphs.append(p_text)

            if not paragraphs and not heading:
                continue

            text = " ".join(paragraphs)
            if not text:
                continue

            section_type = classify_section(heading)

            # Check if this is a subsection (has parent sec)
            parent_sec = None
            for parent in body.iter():
                for child in parent:
                    if child is sec and parent.tag == "sec":
                        parent_title = parent.find("title")
                        if parent_title is not None:
                            parent_sec = _get_text(parent_title)
                        break

            sections.append({
                "heading": heading,
                "section_type": section_type,
                "parent_heading": parent_sec or "",
                "text": text,
                "char_count": len(text),
            })

    # If no structured sections, try to get body text as a single block
    if not sections and body is not None:
        body_text = _clean_text(_get_text(body))
        if body_text and len(body_text) > 50:
            sections.append({
                "heading": "",
                "section_type": "body",
                "parent_heading": "",
                "text": body_text,
                "char_count": len(body_text),
            })

    # Figure captions
    figure_captions = []
    for fig in article.findall(".//fig"):
        fig_id = fig.get("id", "")
        label = fig.findtext("label", default="")
        caption_el = fig.find("caption")
        caption = ""
        if caption_el is not None:
            caption = _clean_text(_get_text(caption_el))
        if caption and len(caption) > 10:
            figure_captions.append({
                "figure_id": fig_id,
                "label": label,
                "caption": caption,
            })

    # Table text
    table_texts = []
    for tw in article.findall(".//table-wrap"):
        table_id = tw.get("id", "")
        label = tw.findtext("label", default="")
        caption_el = tw.find("caption")
        caption = ""
        if caption_el is not None:
            caption = _clean_text(_get_text(caption_el))

        # Try to extract table body text
        body_text = ""
        table_el = tw.find(".//table")
        if table_el is not None:
            # Extract cell text from table
            cells = []
            for cell in table_el.findall(".//td"):
                cell_text = _get_text(cell).strip()
                if cell_text:
                    cells.append(cell_text)
            for cell in table_el.findall(".//th"):
                cell_text = _get_text(cell).strip()
                if cell_text:
                    cells.append(cell_text)
            body_text = " | ".join(cells)

        if (caption and len(caption) > 10) or (body_text and len(body_text) > 20):
            table_texts.append({
                "table_id": table_id,
                "label": label,
                "caption": caption,
                "body_text": body_text[:2000],  # Cap table text
            })

    # Build document
    doc = {
        "pmcid": pmcid,
        "title": _clean_text(title),
        "abstract": _clean_text(abstract),
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

    return doc


def main():
    parser = argparse.ArgumentParser(description="Parse PMC XML into structured docs")
    parser.add_argument("--max", type=int, default=None, help="Max files to parse")
    args = parser.parse_args()

    # Input
    fulltext_dir = get_data_dir("raw/fulltext")
    xml_files = sorted(fulltext_dir.glob("*.xml"))

    if not xml_files:
        print(f"No XML files found in {fulltext_dir}")
        print("Run scripts/02_download_text.py first.")
        sys.exit(1)

    if args.max:
        xml_files = xml_files[: args.max]

    print(f"Parsing {len(xml_files)} PMC XML files...")

    # Output
    output_dir = get_data_dir("interim/parsed")
    output_path = output_dir / "parsed_documents.jsonl"

    t0 = time.time()
    parsed_count = 0
    failed_count = 0
    stats = {
        "section_types": {},
        "has_methods": 0,
        "has_results": 0,
        "total_figures": 0,
        "total_tables": 0,
        "total_sections": 0,
        "char_counts": [],
    }

    with open(output_path, "w") as f:
        for i, xml_file in enumerate(xml_files):
            doc = parse_pmc_xml(xml_file)
            if doc:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                parsed_count += 1

                # Track stats
                for sec in doc["sections"]:
                    st = sec["section_type"]
                    stats["section_types"][st] = stats["section_types"].get(st, 0) + 1
                stats["has_methods"] += doc["has_methods"]
                stats["has_results"] += doc["has_results"]
                stats["total_figures"] += doc["figure_count"]
                stats["total_tables"] += doc["table_count"]
                stats["total_sections"] += doc["section_count"]
                stats["char_counts"].append(doc["total_chars"])
            else:
                failed_count += 1

            done = i + 1
            if done % 200 == 0 or done == len(xml_files):
                print(f"  [{done}/{len(xml_files)}] parsed={parsed_count} failed={failed_count}")

    t1 = time.time()

    # Compute final stats
    char_counts = stats.pop("char_counts")
    if char_counts:
        stats["median_chars"] = sorted(char_counts)[len(char_counts) // 2]
        stats["mean_chars"] = sum(char_counts) // len(char_counts)
        stats["min_chars"] = min(char_counts)
        stats["max_chars"] = max(char_counts)

    stats["parsed_count"] = parsed_count
    stats["failed_count"] = failed_count
    stats["avg_sections_per_doc"] = round(stats["total_sections"] / max(parsed_count, 1), 1)
    stats["avg_figures_per_doc"] = round(stats["total_figures"] / max(parsed_count, 1), 1)
    stats["avg_tables_per_doc"] = round(stats["total_tables"] / max(parsed_count, 1), 1)

    stats_path = output_dir / "parsing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"PARSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Parsed: {parsed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Time: {t1-t0:.0f}s")
    print(f"  Output: {output_path}")
    print(f"\n  Section types:")
    for st, count in sorted(stats["section_types"].items(), key=lambda x: -x[1]):
        print(f"    {count:>5}  {st}")
    print(f"\n  Has methods section: {stats['has_methods']}")
    print(f"  Has results section: {stats['has_results']}")
    print(f"  Avg sections/doc: {stats['avg_sections_per_doc']}")
    print(f"  Avg figures/doc: {stats['avg_figures_per_doc']}")
    print(f"  Avg tables/doc: {stats['avg_tables_per_doc']}")
    if char_counts:
        print(f"  Median doc length: {stats['median_chars']:,} chars")
        print(f"  Mean doc length: {stats['mean_chars']:,} chars")


if __name__ == "__main__":
    main()
