#!/usr/bin/env python3
"""Script 02: Download PMC Open Access full text.

Downloads full-text XML from PMC for papers that have a PMCID.
Uses NCBI E-utilities efetch with rate limiting.

Usage:
    python scripts/02_download_text.py [--max N] [--skip-existing]

    --max N          Only download N papers (for testing)
    --skip-existing  Skip papers already downloaded
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Bio import Entrez
from src.utils.config import get_data_dir

# Entrez config
Entrez.email = "synthsearch@research.example.com"
_api_key = os.environ.get("NCBI_API_KEY")
if _api_key:
    Entrez.api_key = _api_key

_RATE_LIMIT = 10 if _api_key else 3
_MIN_INTERVAL = 1.0 / _RATE_LIMIT
_last_request = 0.0


def _rate_limit():
    global _last_request
    now = time.time()
    elapsed = now - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request = time.time()


def download_pmc_xml(pmcid: str) -> bytes | None:
    """Download full-text XML for a single PMC article.

    Args:
        pmcid: PMC ID (e.g., 'PMC11119143'). Strips 'PMC' prefix if present.

    Returns:
        Raw XML bytes, or None if download failed.
    """
    # Strip 'PMC' prefix for the API call
    numeric_id = pmcid.replace("PMC", "")
    _rate_limit()

    try:
        handle = Entrez.efetch(
            db="pmc",
            id=numeric_id,
            rettype="xml",
            retmode="xml",
        )
        xml_data = handle.read()
        handle.close()
        return xml_data
    except Exception as e:
        print(f"  Warning: failed to download {pmcid}: {e}")
        return None


def download_pmc_xml_batch(pmcids: list[str], batch_size: int = 25) -> dict[str, bytes]:
    """Download full-text XML for a batch of PMC articles.

    Args:
        pmcids: List of PMC IDs.
        batch_size: Number of articles per API call.

    Returns:
        Dict mapping PMCID -> XML bytes.
    """
    results = {}

    for i in range(0, len(pmcids), batch_size):
        batch = pmcids[i : i + batch_size]
        numeric_ids = [p.replace("PMC", "") for p in batch]
        _rate_limit()

        try:
            handle = Entrez.efetch(
                db="pmc",
                id=",".join(numeric_ids),
                rettype="xml",
                retmode="xml",
            )
            xml_data = handle.read()
            handle.close()

            # PMC batch returns a single XML with multiple articles inside
            # We store the whole batch and split later in parsing
            # For simplicity, store per-batch
            for pmcid in batch:
                results[pmcid] = xml_data

        except Exception as e:
            print(f"  Warning: batch download failed ({len(batch)} articles): {e}")
            # Fall back to individual downloads
            for pmcid in batch:
                xml = download_pmc_xml(pmcid)
                if xml:
                    results[pmcid] = xml

        done = min(i + batch_size, len(pmcids))
        if done % 100 == 0 or done == len(pmcids):
            print(f"  Downloaded {done}/{len(pmcids)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download PMC full text")
    parser.add_argument("--max", type=int, default=None, help="Max papers to download")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already downloaded")
    args = parser.parse_args()

    # Load metadata
    metadata_path = get_data_dir("raw/metadata") / "pubmed_openalex_metadata.jsonl"
    if not metadata_path.exists():
        print(f"Error: metadata not found at {metadata_path}")
        print("Run scripts/01_collect_metadata.py first.")
        sys.exit(1)

    # Collect PMCIDs
    pmcid_to_pmid = {}
    with open(metadata_path) as f:
        for line in f:
            rec = json.loads(line)
            pmcid = rec.get("pmcid", "")
            if pmcid:
                pmcid_to_pmid[pmcid] = rec["pmid"]

    print(f"Found {len(pmcid_to_pmid)} papers with PMC IDs")

    # Output directory
    output_dir = get_data_dir("raw/fulltext")

    # Filter already downloaded
    pmcids = list(pmcid_to_pmid.keys())
    if args.skip_existing:
        existing = {f.stem for f in output_dir.glob("*.xml")}
        pmcids = [p for p in pmcids if p not in existing]
        print(f"  {len(existing)} already downloaded, {len(pmcids)} remaining")

    # Cap if requested
    if args.max:
        pmcids = pmcids[: args.max]
        print(f"  Capped to {len(pmcids)} papers")

    if not pmcids:
        print("Nothing to download.")
        return

    print(f"\nDownloading {len(pmcids)} full-text articles from PMC...")
    print(f"  Output: {output_dir}")
    print(f"  Rate limit: {_RATE_LIMIT} req/sec")
    est_time = len(pmcids) / _RATE_LIMIT
    print(f"  Estimated time: {est_time:.0f}s ({est_time/60:.1f}min)")
    print()

    # Download individually (most reliable for parsing later)
    t0 = time.time()
    success = 0
    failed = 0
    failed_ids = []

    for i, pmcid in enumerate(pmcids):
        output_path = output_dir / f"{pmcid}.xml"

        if args.skip_existing and output_path.exists():
            success += 1
            continue

        xml_data = download_pmc_xml(pmcid)
        if xml_data:
            with open(output_path, "wb") as f:
                f.write(xml_data)
            success += 1
        else:
            failed += 1
            failed_ids.append(pmcid)

        done = i + 1
        if done % 50 == 0 or done == len(pmcids):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(pmcids) - done) / rate if rate > 0 else 0
            print(f"  [{done}/{len(pmcids)}] success={success} failed={failed} "
                  f"rate={rate:.1f}/s ETA={remaining:.0f}s")

    t1 = time.time()

    # Save download log
    log = {
        "total_attempted": len(pmcids),
        "success": success,
        "failed": failed,
        "failed_ids": failed_ids,
        "elapsed_seconds": round(t1 - t0, 1),
    }
    log_path = output_dir / "download_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Attempted: {len(pmcids)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Time: {t1-t0:.0f}s")
    print(f"  Output: {output_dir}")
    print(f"  Log: {log_path}")

    if failed_ids:
        print(f"\n  Failed PMCIDs (first 10): {failed_ids[:10]}")


if __name__ == "__main__":
    main()
