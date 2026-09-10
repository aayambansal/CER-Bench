#!/usr/bin/env python3
"""Build anonymous, deterministic ICLR 2027 source and review-supplement archives."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr2027_submission"
DIST = PAPER / "dist"
FIXED_TIME = (2026, 8, 26, 12, 0, 0)
FORBIDDEN_TEXT = (
    "aayambansal",
    "/Users/",
    "prj_9a717f86995f4698a326dc5d4a754456",
    "ses_fbfc24131ffeFE8aBLaa5EOAJ0",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
)


def sanitize_task(row: dict) -> dict:
    keep = (
        "task_id",
        "question",
        "task_family",
        "difficulty",
        "required_constraints",
        "expected_answer_type",
        "supporting_doc_ids",
        "hard_negative_doc_ids",
        "generation_method",
        "verification_status",
        "cluster_size",
    )
    return {key: row[key] for key in keep if key in row}


def sanitize_jsonl(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as src,
        destination.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if line.strip():
                dst.write(
                    json.dumps(sanitize_task(json.loads(line)), ensure_ascii=False)
                    + "\n"
                )


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if (
            not path.is_file()
            or path.name in {".DS_Store"}
            or "__pycache__" in path.parts
        ):
            continue
        copy(path, destination / path.relative_to(source))


def scan_anonymity(root: Path) -> None:
    problems = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {
            ".pkl",
            ".pdf",
            ".png",
            ".zip",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN_TEXT:
            if token.lower() in text.lower():
                problems.append(f"{path.relative_to(root)} contains {token!r}")
    if problems:
        raise RuntimeError("Anonymity scan failed:\n" + "\n".join(problems))


def write_zip(source_root: Path, archive: Path, prefix: str) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(prefix) / path.relative_to(source_root)
            info = zipfile.ZipInfo(relative.as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.suffix == ".sh" else 0o644) << 16
            zf.writestr(info, path.read_bytes())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_source(stage: Path) -> None:
    files = (
        "cerbench_iclr2027.tex",
        "refs.bib",
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
        "natbib.sty",
        "fancyhdr.sty",
        "README.md",
        "SUBMISSION_CHECKLIST.md",
    )
    for name in files:
        copy(PAPER / name, stage / name)
    for name in (
        "task_taxonomy_redesign.png",
        "agent_architecture_redesign.png",
        "label_sensitivity_analysis.pdf",
    ):
        copy(PAPER / "figures" / name, stage / "figures" / name)


def build_supplement(stage: Path) -> None:
    # Sanitized benchmark metadata: no supporting passages, raw abstracts, or reference answers.
    for split in ("train", "dev", "test"):
        sanitize_jsonl(
            ROOT / "data" / "benchmark" / f"{split}.jsonl",
            stage / "data" / "benchmark" / f"{split}.jsonl",
        )
    for name in ("tasks_v2.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl"):
        sanitize_jsonl(
            ROOT / "data" / "benchmark" / "v2" / name,
            stage / "data" / "benchmark" / "v2" / name,
        )
    copy(
        ROOT / "data" / "benchmark" / "v2" / "agent_test.jsonl",
        stage / "data" / "benchmark" / "v2" / "agent_test.jsonl",
    )
    copy(
        ROOT / "data" / "benchmark" / "schema.json",
        stage / "data" / "benchmark" / "schema.json",
    )
    copy(
        ROOT / "data" / "processed" / "corpus_stats.json",
        stage / "data" / "processed" / "corpus_stats.json",
    )
    copy(
        ROOT / "data" / "processed" / "indices" / "bm25" / "bm25_index.pkl",
        stage / "data" / "processed" / "indices" / "bm25" / "bm25_index.pkl",
    )
    copy(
        ROOT / "data" / "processed" / "indices" / "bm25" / "chunk_ids.json",
        stage / "data" / "processed" / "indices" / "bm25" / "chunk_ids.json",
    )

    result_names = (
        "bm25_test.jsonl",
        "dense_test.jsonl",
        "bge_test.jsonl",
        "e5large_test.jsonl",
        "medcpt_test.jsonl",
        "splade_test.jsonl",
        "rm3_test.jsonl",
        "hybrid_test.jsonl",
        "hybrid_reranked_test.jsonl",
        "bge_reranker_test.jsonl",
        "agent_single_step_test.jsonl",
        "agent_test.jsonl",
        "gold_adjudication.jsonl",
        "expanded_gold.json",
        "scores_test_adjudicated.json",
        "scores_test_comprehensive.json",
        "abstention_metrics.json",
    )
    for name in result_names:
        copy(
            ROOT / "results" / "baselines" / name,
            stage / "results" / "baselines" / name,
        )
    copy(
        ROOT / "results" / "paper_tables" / "submission_audit.json",
        stage / "results" / "paper_tables" / "submission_audit.json",
    )

    copy_tree(ROOT / "scripts", stage / "scripts")
    # The local package builder embeds anonymity-test sentinel strings and is
    # therefore intentionally not shipped inside the anonymous supplement.
    (stage / "scripts" / "33_build_review_packages.py").unlink()
    copy_tree(ROOT / "src", stage / "src")
    copy_tree(ROOT / "configs", stage / "configs")
    copy(ROOT / "requirements.txt", stage / "requirements-full.txt")
    (stage / "requirements-review.txt").write_text(
        "numpy>=1.24\nmatplotlib>=3.8\nPillow>=10\nrank-bm25>=0.2.2\n",
        encoding="utf-8",
    )

    paper_stage = stage / "paper" / "iclr2027_submission"
    build_source(paper_stage)
    copy(PAPER / "REFERENCE_AUDIT.md", paper_stage / "REFERENCE_AUDIT.md")
    copy(PAPER / "reference_validation.json", paper_stage / "reference_validation.json")

    (stage / "README.md").write_text(
        """# CER-Bench anonymous ICLR 2027 review supplement

This archive is the evidence bundle for the anonymous submission. It contains sanitized task metadata, saved document-ID retrieval outputs, qrel summaries, the exact BM25 index needed for the v2 audit, analysis/build scripts, implementation source, configs, and the paper source.

It deliberately excludes raw PubMed abstracts, PMC XML/full text, supporting text passages, credentials, caches, model weights, and author-identifying metadata. The benchmark files retain questions, constraints, split/family labels, supporting document identifiers, and hard-negative identifiers.

## Lightweight verification

```sh
python -m pip install -r requirements-review.txt
python scripts/31_submission_audit.py
python scripts/32_make_submission_figure.py
```

If a full LaTeX installation is available:

```sh
bash scripts/run_iclr2027_submission.sh
```

Commercial-API task generation, LLM adjudication, neural index construction, and agent inference are not rerun by this bundle; the audit verifies paper claims against their saved outputs. See the manuscript's reproducibility and data-governance appendices for scope and limitations.
""",
        encoding="utf-8",
    )


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/45_submission_readiness.py"), "--require-ready"],
        check=True,
    )
    if not (PAPER / "cerbench_iclr2027.pdf").exists():
        raise FileNotFoundError("Build cerbench_iclr2027.pdf before packaging")
    DIST.mkdir(parents=True, exist_ok=True)
    if any(DIST.glob("CERBench_ICLR2027*")):
        raise FileExistsError("Historical dist artifacts are preserved; use a versioned release packager after clearance.")

    with (
        tempfile.TemporaryDirectory(prefix="cerbench-iclr-source-") as source_tmp,
        tempfile.TemporaryDirectory(prefix="cerbench-iclr-supp-") as supp_tmp,
    ):
        source_stage = Path(source_tmp)
        supp_stage = Path(supp_tmp)
        build_source(source_stage)
        build_supplement(supp_stage)
        scan_anonymity(source_stage)
        scan_anonymity(supp_stage)

        source_zip = DIST / "CERBench_ICLR2027_LaTeX.zip"
        supplement_zip = DIST / "CERBench_ICLR2027_Supplement.zip"
        write_zip(source_stage, source_zip, "CERBench_ICLR2027_LaTeX")
        write_zip(supp_stage, supplement_zip, "CERBench_ICLR2027_Supplement")

    final_pdf = DIST / "CERBench_ICLR2027.pdf"
    copy(PAPER / "cerbench_iclr2027.pdf", final_pdf)
    checklist = DIST / "CERBench_ICLR2027_SUBMISSION_CHECKLIST.md"
    copy(PAPER / "SUBMISSION_CHECKLIST.md", checklist)
    outputs = [final_pdf, source_zip, supplement_zip, checklist]
    manifest = {
        "anonymous": True,
        "main_text_end_page": 7,
        "pdf_pages_total": 14,
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    }
    manifest_path = DIST / "CERBench_ICLR2027_SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
