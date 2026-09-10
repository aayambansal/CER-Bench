#!/usr/bin/env python3
"""Offline authoritative rebuild. Network retrieval MUST be external WebFetch.

--plan prints <=200-ID URLs. --ingest consumes an explicit successful WebFetch
request log and moves new XML files without overwriting. Default builds from
checksummed manifests, never from historical metadata as authority.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.corpus.pubmed_xml import parse_pubmed_xml, inspect_fulltext, normalize_pmc, local_tree, text

RAW = Path("data/raw/pubmed_verified_v1")
OUT = Path("data/processed/authoritative_v1")
REPORT = Path("results/readiness/authoritative_corpus")
URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


def plans(old):
    ids = sorted({str(r["pmid"]) for r in old}, key=int)
    return [dict(filename=f"pubmed_batch_{i // 200 + 1:03d}.xml", requested_pmids=ids[i:i + 200],
                 url=URL + ",".join(ids[i:i + 200]) + "&retmode=xml") for i in range(0, len(ids), 200)]


def ingest(base, scratch, request_log):
    """Require an actual request log, not an inferred list of returned IDs."""
    entries = rows(request_log)
    raw = base / RAW
    raw.mkdir(parents=True, exist_ok=True)
    # Preflight the whole transfer before moving any files.
    for e in entries:
        name = e["filename"]
        if Path(name).name != name or not name.endswith(".xml"):
            raise ValueError("invalid raw filename")
        if e["url"] != URL + ",".join(e["requested_pmids"]) + "&retmode=xml":
            raise ValueError("request URL/IDs disagree")
        if not 1 <= len(e["requested_pmids"]) <= 200:
            raise ValueError("request exceeds batch bound")
        source = scratch / name
        if (raw / name).exists() or (raw / (name + ".manifest.json")).exists():
            raise FileExistsError(name)
        if sha(source) != e["sha256"]:
            raise ValueError("broker checksum mismatch: " + name)
        parse_pubmed_xml(source.read_bytes(), e["requested_pmids"])
    for e in entries:
        source = scratch / e["filename"]
        parsed = parse_pubmed_xml(source.read_bytes(), e["requested_pmids"])
        manifest = dict(e, bytes=source.stat().st_size, transport="WebFetch",
                        returned_pmids=parsed["returned_pmids"],
                        missing_pmids=parsed["missing_pmids"], unexpected_pmids=parsed["unexpected_pmids"],
                        duplicate_pmids=parsed["duplicate_pmids"], problems=parsed["problems"])
        shutil.move(str(source), str(raw / e["filename"]))
        write_json(raw / (e["filename"] + ".manifest.json"), manifest)


def build(base):
    old_path = base / "data/processed/corpus.jsonl"
    old = rows(old_path)
    target = {str(r["pmid"]) for r in old}
    records, provenance, failures, manifests = [], {}, [], []
    cited_ids = {}
    requested, returned = [], []
    for path in sorted((base / RAW).glob("pubmed_batch_*.xml.manifest.json")):
        m = json.loads(path.read_text())
        source = base / RAW / m["filename"]
        manifests.append(m)
        requested.extend(m["requested_pmids"])
        if sha(source) != m["sha256"] or source.stat().st_size != m["bytes"]:
            failures.append(dict(file=str(source.relative_to(base)), reason="checksum_mismatch"))
            continue
        if m["url"] != URL + ",".join(m["requested_pmids"]) + "&retmode=xml":
            failures.append(dict(file=m["filename"], reason="manifest_url_mismatch"))
            continue
        parsed = parse_pubmed_xml(source.read_bytes(), m["requested_pmids"])
        for node in local_tree(source.read_bytes()).findall("PubmedArticle"):
            cited = node.findall("PubmedData/ReferenceList/.//ArticleId")
            cited_ids[text(node.find("MedlineCitation/PMID"))] = {
                "pmcid": {normalize_pmc(text(e)) for e in cited if e.get("IdType") == "pmc"},
                "doi": {text(e) for e in cited if e.get("IdType") == "doi"}}
        if parsed["returned_pmids"] != m["returned_pmids"]:
            failures.append(dict(file=m["filename"], reason="manifest_returned_ids_mismatch"))
            continue
        returned.extend(parsed["returned_pmids"])
        failures.extend(dict(file=m["filename"], **p) for p in parsed["problems"])
        failures.extend(dict(file=m["filename"], pmid=p, reason="duplicate_in_batch") for p in parsed["duplicate_pmids"])
        failures.extend(dict(file=m["filename"], pmid=p, reason="unexpected_return") for p in parsed["unexpected_pmids"])
        records.extend(parsed["records"])
        for r in parsed["records"]:
            provenance[r["pmid"]] = dict(source_url=m["url"], raw_path=str(source.relative_to(base)),
                                        raw_sha256=m["sha256"], retrieved_at_utc=m["retrieved_at_utc"],
                                        retrieval_date=m["retrieved_at_utc"][:10], transport="WebFetch")
    duplicate = sorted(p for p, n in Counter(returned).items() if n > 1)
    failures.extend(dict(pmid=p, reason="duplicate_across_batches") for p in duplicate)
    by_id = {r["pmid"]: r for r in records if r["pmid"] in target and r["pmid"] not in duplicate}
    missing = sorted(target - by_id.keys(), key=int)
    failures.extend(dict(pmid=p, reason="missing_or_unsupported_authoritative_record") for p in missing)
    failures.extend(dict(pmid=p, reason="never_requested") for p in sorted(target - set(requested)))
    if len(old) != len(target):
        failures.append(dict(reason="duplicate_historical_pmid"))

    # Inspect every local XML's front, never use its filename as identity.
    local = []
    for path in sorted((base / "data/raw/fulltext").glob("*.xml")):
        try:
            item = inspect_fulltext(path.read_bytes())
        except Exception as exc:
            item = dict(status="xml_parse_error", error=str(exc), pmids=[], pmcids=[],
                        license_xml=[], redistribution_status="not_assessed_no_rights_granted")
        item.update(path=str(path.relative_to(base)), sha256=sha(path), bytes=path.stat().st_size)
        local.append(item)
    pairs = defaultdict(list)
    for item in local:
        if item["status"] == "front_identity_candidate":
            pairs[(item["pmids"][0], item["pmcids"][0])].append(item)
    pmc_owners = defaultdict(list)
    for r in by_id.values():
        if r["pmcid"]:
            pmc_owners[r["pmcid"]].append(r["pmid"])
    for pmc, owners in pmc_owners.items():
        if len(owners) > 1:
            failures.append(dict(reason="authoritative_pmc_collision", pmcid=pmc, pmids=owners))
    for item in local:
        if item["status"] == "front_identity_candidate":
            p, pmc = item["pmids"][0], item["pmcids"][0]
            item["authoritative_identity_match"] = p in by_id and by_id[p]["pmcid"] == pmc
        else:
            item["authoritative_identity_match"] = False
    # Also require each local front ID to belong to exactly one local file,
    # including files with conflicting IDs (which must not get ignored).
    local_pmid_counts = Counter(p for item in local for p in item["pmids"])
    local_pmc_counts = Counter(p for item in local for p in item["pmcids"])
    docs, chunks, comparisons, eligibility, absences = [], [], [], [], []
    fields = ("doc_id", "pmcid", "doi", "title", "abstract", "year", "venue", "venue_abbrev",
              "authors", "mesh_terms", "publication_types", "keywords")
    change_counts = Counter()
    fulltext_counts = Counter()
    old_by_pmid = {str(r["pmid"]): r for r in old}
    for pmid in sorted(target, key=int):
        prior = old_by_pmid[pmid]
        if pmid not in by_id:
            comparisons.append(dict(pmid=pmid, status="missing_or_unsupported", changes={}))
            eligibility.append(dict(pmid=pmid, legacy_doc_id=prior["doc_id"], eligible_for_new_task_generation=False,
                                    eligible_as_gold=False, reasons=["missing_authoritative_record", "historical_gold_not_reused"]))
            continue
        r = dict(by_id[pmid], doc_id=pmid, provenance=provenance[pmid])
        matches = pairs.get((pmid, r["pmcid"]), [])
        status = "no_authoritative_pmc" if not r["pmcid"] else "no_matching_local_front"
        if matches:
            if len(matches) == 1 and len(pmc_owners[r["pmcid"]]) == 1 and local_pmid_counts[pmid] == 1 and local_pmc_counts[r["pmcid"]] == 1:
                status = "identity_matched_raw_candidate_text_unvalidated"
            else:
                status = "ambiguous_local_or_authoritative_identity"
        fulltext_counts[status] += 1
        r.update(has_fulltext=False, sections=[], figure_captions=[], table_texts=[],
                 fulltext_status=status, fulltext_candidates=[dict(m) for m in matches],
                 redistribution_status="not_assessed_no_rights_granted",
                 human_validation="waived_not_done", biomedical_relevance_verified=False)
        absent = [f for f in ("pmcid", "doi", "title", "abstract", "year") if not r[f]]
        if absent:
            absences.append(dict(pmid=pmid, absent_fields=absent,
                                 reason="absent_in_authoritative_article_metadata_not_imputed"))
        if not r["title"]:
            failures.append(dict(pmid=pmid, reason="missing_required_title"))
        changes = {f: dict(old=prior.get(f), authoritative=r.get(f)) for f in fields if prior.get(f) != r.get(f)}
        change_counts.update(changes.keys())
        comparisons.append(dict(pmid=pmid, status="compared", changes=changes,
                                changed_old_ids_found_in_current_references=[f for f in ("pmcid", "doi")
                                    if f in changes and prior.get(f) and
                                    (normalize_pmc(prior[f]) if f == "pmcid" else prior[f]) in cited_ids.get(pmid, {}).get(f, set())],
                                old_has_fulltext=prior.get("has_fulltext", False), new_fulltext_status=status))
        docs.append(r)
        if r["abstract"]:
            chunks.append(dict(chunk_id=f"{pmid}_abstract_0", doc_id=pmid, pmid=pmid, pmcid=r["pmcid"],
                               year=r["year"], venue=r["venue"], mesh_terms=r["mesh_terms"], seed_query="",
                               section_type="abstract", section_heading="Abstract", text=r["abstract"],
                               token_estimate=len(r["abstract"].split()), position=0,
                               provenance=r["provenance"], text_source="authoritative_pubmed_abstract"))
        reasons = ["historical_tasks_and_qrels_not_reused", "human_validation_waived_not_done",
                   "identity_is_not_biomedical_relevance"]
        if not r["abstract"]:
            reasons.append("no_authoritative_abstract")
        eligibility.append(dict(pmid=pmid, legacy_doc_id=prior["doc_id"], authoritative_doc_id=pmid,
                                eligible_for_new_task_generation=bool(r["title"] and r["abstract"]),
                                eligible_as_gold=False, fulltext_status=status, reasons=reasons))

    # Historical task IDs/support IDs only: do not copy questions, answers,
    # passages, labels, or qrels into the new corpus/gold.
    legacy = defaultdict(set)
    for r in old:
        legacy[r["doc_id"]].add(str(r["pmid"]))
    task_mapping, task_inputs = [], []
    task_paths = sorted(set((base / "data/benchmark").rglob("*.jsonl")) |
                        set((base / "data/interim").glob("*tasks.jsonl")) |
                        set((base / "data/training").glob("*tasks.jsonl")))
    for path in task_paths:
        task_inputs.append(dict(path=str(path.relative_to(base)), sha256=sha(path)))
        for line, task in enumerate(rows(path), 1):
            support = task.get("supporting_doc_ids", [])
            task_mapping.append(dict(source_file=str(path.relative_to(base)), source_line=line,
                                     task_id=task.get("task_id", task.get("id")),
                                     historical_support_ids=support,
                                     candidate_pmids_by_legacy_id={s: sorted(legacy.get(s, set())) for s in support},
                                     eligible_as_gold=False, eligible_for_automatic_migration=False,
                                     reasons=["historical_task_and_qrel_contamination_not_excluded",
                                              "regenerate_and_validate_against_authoritative_text",
                                              "human_validation_waived_not_done"]))
    # Direct reproduction of the observed bug, using the pilot as evidence only.
    pilot = base / RAW / "identity_pilot.xml"
    root_cause = []
    if pilot.exists():
        for a in local_tree(pilot.read_bytes()).findall("PubmedArticle"):
            root_cause.append(dict(pmid=text(a.find("MedlineCitation/PMID")),
                                   own_pmc_ids=[text(e) for e in a.findall("PubmedData/ArticleIdList/ArticleId") if e.get("IdType") == "pmc"],
                                   buggy_first_descendant_pmc=next((text(e) for e in a.findall("PubmedData/.//ArticleId") if e.get("IdType") == "pmc"), ""),
                                   cited_pmc6286148=[text(ref.find("Citation")) for ref in a.findall("PubmedData/ReferenceList/.//Reference")
                                                    if any(normalize_pmc(text(e)) == "PMC6286148" for e in ref.findall("ArticleIdList/ArticleId") if e.get("IdType") == "pmc")],
                                   pilot_url=URL + "38308006,30610625,40828286&retmode=xml", pilot_sha256=sha(pilot)))
    summary = dict(schema_version="authoritative_v1", source="PubMed EFetch XML via WebFetch",
                   historical_rows=len(old), unique_target_pmids=len(target), batches=len(manifests),
                   request_id_occurrences=len(requested), unique_requested_ids=len(set(requested)),
                   returned_records=len(returned), parsed_rows=len(docs), authoritative_pmc_rows=sum(bool(r["pmcid"]) for r in docs),
                   authoritative_doi_rows=sum(bool(r["doi"]) for r in docs), abstract_chunks=len(chunks),
                   restored_fulltext_rows=0, local_xml_files=len(local), fulltext_status_counts=dict(fulltext_counts),
                   compared_rows=sum(r["status"] == "compared" for r in comparisons),
                   changed_rows=sum(bool(r["changes"]) for r in comparisons), change_counts=dict(change_counts),
                   metadata_changed_rows_excluding_doc_id=sum(bool(set(r["changes"]) - {"doc_id"}) for r in comparisons),
                   changed_old_ids_found_in_current_references=dict(Counter(f for r in comparisons for f in r.get("changed_old_ids_found_in_current_references", []))),
                   pmcid_change_types=dict(Counter("replaced" if c["old"] and c["authoritative"] else "removed" if c["old"] else "added"
                                                for r in comparisons if (c := r["changes"].get("pmcid")))),
                   local_xml_status_counts=dict(Counter(r["status"] for r in local)),
                   local_xml_with_permissions=sum(bool(r["license_xml"]) for r in local),
                   old_fulltext_rows=sum(bool(r.get("has_fulltext")) for r in old),
                   absent_field_counts=dict(Counter(f for r in absences for f in r["absent_fields"])),
                   missing_or_unsupported_pmids=missing, failures=failures,
                   metadata_release_pass=not failures and len(docs) == len(target) and bool(target),
                   benchmark_gold_release_pass=False, human_validation="waived_not_done",
                   external_identity_verified=not failures and len(docs) == len(target) and bool(target),
                   biomedical_relevance_verified=False, public_redistribution_rights_verified=False,
                   historical_gold_reused=False, historical_task_mapping_rows=len(task_mapping),
                   download_http_failures=[],
                   request_anomalies=[dict(filename=m["filename"], missing_pmids=m["missing_pmids"],
                                           note=m.get("request_note", "")) for m in manifests
                                      if m["missing_pmids"] or m.get("request_note")],
                   raw_bytes=sum(m["bytes"] for m in manifests),
                   retrieval_dates=sorted({m["retrieved_at_utc"][:10] for m in manifests}),
                   root_cause_evidence=root_cause,
                   historical_corpus_sha256=sha(old_path), task_inputs=task_inputs,
                   implementation_sha256={p: sha(base / p) for p in ("scripts/47_build_authoritative_corpus.py", "src/corpus/pubmed_xml.py") if (base / p).exists()})
    for name, values in [("corpus.jsonl", docs), ("chunks.jsonl", chunks),
                         ("document_task_eligibility.jsonl", eligibility), ("historical_task_eligibility.jsonl", task_mapping)]:
        write_rows(base / OUT / name, values)
    for name, values in [("metadata_comparison.jsonl", comparisons), ("local_fulltext_audit.jsonl", local),
                         ("metadata_absences.jsonl", absences), ("missing_unsupported_records.jsonl", failures)]:
        write_rows(base / REPORT / name, values)
    write_json(base / REPORT / "summary.json", summary)
    output_files = sorted((base / OUT).glob("*.jsonl")) + sorted((base / REPORT).glob("*.jsonl")) + [base / REPORT / "summary.json"]
    write_json(base / REPORT / "output_manifest.json", [dict(path=str(p.relative_to(base)), sha256=sha(p), bytes=p.stat().st_size) for p in output_files])
    print(json.dumps({k: v for k, v in summary.items() if k not in ("task_inputs", "root_cause_evidence")}, indent=2))
    return 0 if summary["metadata_release_pass"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--ingest", type=Path, help="scratch directory containing successful WebFetch XML downloads")
    parser.add_argument("--request-log", type=Path)
    args = parser.parse_args()
    if args.plan:
        for p in plans(rows(args.base / "data/processed/corpus.jsonl")):
            print(json.dumps(p))
        return 0
    if args.ingest:
        if not args.request_log:
            parser.error("--ingest requires --request-log")
        ingest(args.base, args.ingest, args.request_log)
    return build(args.base)


if __name__ == "__main__":
    sys.exit(main())
