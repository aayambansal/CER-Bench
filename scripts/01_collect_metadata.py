#!/usr/bin/env python3
"""Script 01: Collect metadata from PubMed and OpenAlex.

This script collects paper metadata for the SynthSearch-Biomed corpus.
Strategy:
  1. Run seed PubMed queries across target subdomains
  2. Fetch full PubMed metadata (title, abstract, MeSH, authors, etc.)
  3. Enrich with OpenAlex data (citation graph, concepts, topics, OA status)
  4. Merge records and deduplicate by PMID
  5. Save to data/raw/metadata/

Usage:
    python scripts/01_collect_metadata.py [--pilot] [--max-per-query N]

    --pilot          Collect a small pilot corpus (~5K papers)
    --max-per-query  Max papers per seed query (default: 5000 for pilot, 25000 for full)
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.pubmed_client import search_pubmed, fetch_pubmed_records
from src.corpus.openalex_client import get_works_by_pmids, parse_work
from src.utils.config import load_config, get_data_dir


# ─── Seed queries ────────────────────────────────────────────────────────────
# These queries target subdomains rich in experimental conditions, methods
# comparisons, and condition-dependent findings — ideal for all 8 task families.
#
# Each query is designed to return papers where:
# - Multiple organisms/cell lines are studied (constraint, comparative)
# - Multiple methods are compared (comparative, aggregation)
# - Contradictory results exist under different conditions (contradiction)
# - Quantitative measurements are reported (aggregation)
# - Some combinations are under-studied (abstention)
# - Evidence chains through mechanisms exist (multi-hop)
# - Understanding has evolved over time (temporal)
# - Negative results are reported (negative)

SEED_QUERIES = [
    # Gene editing methods (CRISPR variants, conditions, organisms)
    {
        "name": "crispr_methods",
        "query": (
            "(CRISPR OR gene editing OR base editing OR prime editing) "
            "AND (methods OR protocol OR efficiency OR delivery) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Immune checkpoint therapy (multiple cancer types, combinations, biomarkers)
    {
        "name": "checkpoint_immunotherapy",
        "query": (
            "(immune checkpoint inhibitor OR anti-PD-1 OR anti-PD-L1 OR anti-CTLA-4 "
            "OR pembrolizumab OR nivolumab OR atezolizumab) "
            "AND (cancer OR tumor OR neoplasm) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Drug repurposing (metformin, statins — rich in contradictory findings)
    {
        "name": "drug_repurposing",
        "query": (
            "(drug repurposing OR drug repositioning) "
            "AND (cancer OR neurodegeneration OR Alzheimer OR Parkinson) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Cell therapy / CAR-T (multiple designs, targets, conditions)
    {
        "name": "car_t_cell_therapy",
        "query": (
            "(CAR-T OR CAR T cell OR chimeric antigen receptor) "
            "AND (therapy OR treatment OR clinical) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Organoids and disease modeling (multiple tissue types, conditions)
    {
        "name": "organoid_models",
        "query": (
            "(organoid OR organoids) "
            "AND (disease model OR drug screening OR patient-derived) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Single-cell sequencing methods (technology comparisons)
    {
        "name": "single_cell_methods",
        "query": (
            "(single-cell RNA-seq OR scRNA-seq OR single-cell analysis OR single cell transcriptomics) "
            "AND (method OR benchmark OR comparison OR pipeline) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Microbiome and disease (gut-brain, cancer, metabolic — evolving field)
    {
        "name": "microbiome_disease",
        "query": (
            "(gut microbiome OR intestinal microbiota OR microbiome composition) "
            "AND (cancer OR immunotherapy OR metabolic disease OR diabetes) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # mRNA therapeutics (beyond vaccines — rapidly evolving)
    {
        "name": "mrna_therapeutics",
        "query": (
            "(mRNA therapeutic OR mRNA vaccine OR mRNA delivery OR lipid nanoparticle) "
            "AND (therapy OR treatment OR clinical trial) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Protein structure / AlphaFold impact (temporal evolution)
    {
        "name": "protein_structure_prediction",
        "query": (
            "(AlphaFold OR protein structure prediction OR protein folding) "
            "AND (drug discovery OR drug design OR virtual screening) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
    # Epigenetics and disease (condition-dependent, multi-hop)
    {
        "name": "epigenetics_disease",
        "query": (
            "(epigenetics OR DNA methylation OR histone modification OR chromatin remodeling) "
            "AND (cancer OR neurodegeneration OR development) "
            "AND Journal Article[pt] AND English[la]"
        ),
    },
]


def collect_pubmed_metadata(
    queries: list[dict],
    max_per_query: int,
    min_date: str,
    max_date: str,
    output_dir: Path,
) -> dict[str, dict]:
    """Run PubMed queries and fetch metadata.

    Returns:
        Dict mapping PMID -> record.
    """
    all_records: dict[str, dict] = {}
    query_stats = {}

    for q in queries:
        name = q["name"]
        query = q["query"]
        print(f"\n{'='*60}")
        print(f"Query: {name}")
        print(f"{'='*60}")

        # Search
        print(f"  Searching PubMed...")
        pmids = search_pubmed(
            query=query,
            max_results=max_per_query,
            min_date=min_date,
            max_date=max_date,
        )
        print(f"  Found {len(pmids)} PMIDs")

        # Remove already-fetched
        new_pmids = [p for p in pmids if p not in all_records]
        print(f"  {len(new_pmids)} new (after dedup)")

        if not new_pmids:
            query_stats[name] = {"searched": len(pmids), "new": 0, "fetched": 0}
            continue

        # Fetch records
        print(f"  Fetching metadata for {len(new_pmids)} papers...")
        records = fetch_pubmed_records(new_pmids)
        print(f"  Fetched {len(records)} records")

        # Index by PMID
        for rec in records:
            rec["seed_query"] = name
            all_records[rec["pmid"]] = rec

        query_stats[name] = {
            "searched": len(pmids),
            "new": len(new_pmids),
            "fetched": len(records),
        }

    # Save query stats
    stats_path = output_dir / "query_stats.json"
    with open(stats_path, "w") as f:
        json.dump(query_stats, f, indent=2)
    print(f"\nQuery stats saved to {stats_path}")

    return all_records


def enrich_with_openalex(
    records: dict[str, dict],
    output_dir: Path,
) -> dict[str, dict]:
    """Enrich PubMed records with OpenAlex data (citations, concepts, topics).

    Returns:
        Updated records dict with OpenAlex fields merged in.
    """
    pmids = list(records.keys())
    print(f"\nEnriching {len(pmids)} records with OpenAlex data...")

    oa_works = get_works_by_pmids(pmids)
    print(f"  Found {len(oa_works)} matches in OpenAlex")

    enriched_count = 0
    for pmid, oa_work in oa_works.items():
        if pmid in records:
            parsed = parse_work(oa_work)
            rec = records[pmid]

            # Merge OpenAlex fields (don't overwrite PubMed core fields)
            rec["openalex_id"] = parsed.get("openalex_id", "")
            rec["concepts"] = parsed.get("concepts", [])
            rec["topics"] = parsed.get("topics", [])
            rec["cited_by_count"] = parsed.get("cited_by_count", 0)
            rec["referenced_work_ids"] = parsed.get("referenced_work_ids", [])
            rec["is_oa"] = parsed.get("is_oa", False)
            rec["oa_url"] = parsed.get("oa_url", "")

            # Fill in PMC ID from OpenAlex if PubMed didn't have it
            if not rec.get("pmcid") and parsed.get("pmcid"):
                rec["pmcid"] = parsed["pmcid"]

            # Fill abstract from OpenAlex if PubMed didn't have one
            if not rec.get("abstract") and parsed.get("abstract"):
                rec["abstract"] = parsed["abstract"]

            enriched_count += 1

    print(f"  Enriched {enriched_count} records")
    return records


def compute_stats(records: dict[str, dict]) -> dict:
    """Compute summary statistics for the collected metadata."""
    total = len(records)
    recs = list(records.values())

    years = [r.get("year") for r in recs if r.get("year")]
    has_abstract = sum(1 for r in recs if r.get("abstract"))
    has_pmcid = sum(1 for r in recs if r.get("pmcid"))
    has_openalex = sum(1 for r in recs if r.get("openalex_id"))
    has_mesh = sum(1 for r in recs if r.get("mesh_terms"))
    has_doi = sum(1 for r in recs if r.get("doi"))
    is_oa = sum(1 for r in recs if r.get("is_oa"))

    # Year distribution
    year_counts = Counter(years)

    # Top venues
    venue_counts = Counter(r.get("venue", "") for r in recs if r.get("venue"))
    top_venues = venue_counts.most_common(20)

    # Publication type distribution
    pub_types = Counter()
    for r in recs:
        for pt in r.get("publication_types", []):
            pub_types[pt] += 1

    # Seed query distribution
    query_counts = Counter(r.get("seed_query", "unknown") for r in recs)

    # MeSH term frequency
    mesh_counts = Counter()
    for r in recs:
        for m in r.get("mesh_terms", []):
            mesh_counts[m] += 1

    stats = {
        "total_records": total,
        "has_abstract": has_abstract,
        "has_abstract_pct": round(has_abstract / total * 100, 1) if total else 0,
        "has_pmcid": has_pmcid,
        "has_pmcid_pct": round(has_pmcid / total * 100, 1) if total else 0,
        "has_openalex": has_openalex,
        "has_openalex_pct": round(has_openalex / total * 100, 1) if total else 0,
        "has_mesh": has_mesh,
        "has_doi": has_doi,
        "is_open_access": is_oa,
        "year_range": [min(years), max(years)] if years else [],
        "year_distribution": dict(sorted(year_counts.items())),
        "top_venues": top_venues,
        "publication_types": pub_types.most_common(20),
        "seed_query_distribution": dict(query_counts),
        "top_mesh_terms": mesh_counts.most_common(50),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect metadata from PubMed + OpenAlex")
    parser.add_argument("--pilot", action="store_true", help="Collect pilot corpus (~5K papers)")
    parser.add_argument("--max-per-query", type=int, default=None, help="Max papers per query")
    args = parser.parse_args()

    # Load config
    config = load_config("corpus")
    min_year = config["filtering"]["year_range"]["min"]
    max_year = config["filtering"]["year_range"]["max"]

    # Set limits
    if args.pilot:
        max_per_query = args.max_per_query or 1000
        mode = "PILOT"
    else:
        max_per_query = args.max_per_query or 10000
        mode = "FULL"

    print(f"{'='*60}")
    print(f"SynthSearch Metadata Collection ({mode})")
    print(f"{'='*60}")
    print(f"  Seed queries: {len(SEED_QUERIES)}")
    print(f"  Max per query: {max_per_query}")
    print(f"  Year range: {min_year}-{max_year}")
    print(f"  Estimated papers: {len(SEED_QUERIES) * max_per_query} (before dedup)")
    print()

    # Output directory
    output_dir = get_data_dir("raw/metadata")

    # Step 1: PubMed queries
    t0 = time.time()
    records = collect_pubmed_metadata(
        queries=SEED_QUERIES,
        max_per_query=max_per_query,
        min_date=f"{min_year}/01/01",
        max_date=f"{max_year}/12/31",
        output_dir=output_dir,
    )
    t1 = time.time()
    print(f"\nPubMed collection: {len(records)} unique records in {t1-t0:.0f}s")

    # Step 2: OpenAlex enrichment
    records = enrich_with_openalex(records, output_dir)
    t2 = time.time()
    print(f"OpenAlex enrichment done in {t2-t1:.0f}s")

    # Step 3: Filter
    print(f"\nFiltering records...")
    excluded_types = set(config["filtering"]["exclude_publication_types"])
    filtered = {}
    excluded_count = 0
    no_abstract_count = 0

    for pmid, rec in records.items():
        # Check publication type
        pub_types = set(rec.get("publication_types", []))
        if pub_types & excluded_types:
            excluded_count += 1
            continue

        # Require abstract
        if config["filtering"]["require_abstract"] and not rec.get("abstract"):
            no_abstract_count += 1
            continue

        filtered[pmid] = rec

    print(f"  Excluded {excluded_count} by publication type")
    print(f"  Excluded {no_abstract_count} without abstract")
    print(f"  Remaining: {len(filtered)} records")

    # Step 4: Save
    output_path = output_dir / "pubmed_openalex_metadata.jsonl"
    with open(output_path, "w") as f:
        for rec in filtered.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved to {output_path}")

    # Step 5: Stats
    stats = compute_stats(filtered)
    stats_path = output_dir / "collection_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records: {stats['total_records']}")
    print(f"  With abstract: {stats['has_abstract']} ({stats['has_abstract_pct']}%)")
    print(f"  With PMC ID (full text available): {stats['has_pmcid']} ({stats['has_pmcid_pct']}%)")
    print(f"  OpenAlex enriched: {stats['has_openalex']} ({stats['has_openalex_pct']}%)")
    print(f"  Open access: {stats['is_open_access']}")
    print(f"  Year range: {stats['year_range']}")
    print(f"\n  Top 5 venues:")
    for venue, count in stats["top_venues"][:5]:
        print(f"    {count:>5}  {venue}")
    print(f"\n  Seed query distribution:")
    for query, count in sorted(stats["seed_query_distribution"].items()):
        print(f"    {count:>5}  {query}")

    t3 = time.time()
    print(f"\nTotal runtime: {t3-t0:.0f}s")


if __name__ == "__main__":
    main()
