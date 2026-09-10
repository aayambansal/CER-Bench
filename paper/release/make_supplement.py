#!/usr/bin/env python3
"""Build the anonymized OpenReview supplement from the committed git tree.

Usage (from the repository root):
    python paper/release/make_supplement.py [--out paper/release/supplementary_material.zip]

The supplement is `git archive HEAD` minus named-author material, with author names,
emails, GitHub/Hugging Face usernames, institution, public artifact URLs, and absolute
machine paths replaced by placeholders in every text file. Binary files are copied
unchanged; PDFs that mention the authors are dropped. The result is for double-blind
review only; provenance manifests inside it still hash the *original* paths, so use the
public repository, not this archive, to replay hash-bound verification.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

TEXT_EXT = {".md", ".txt", ".py", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".tex", ".bib",
            ".sty", ".bst", ".cfg", ".toml", ".csv", ".xml", ".log", ".html", ".ini", ".gitignore"}

# Order matters: longer / more specific patterns first.
REPLACEMENTS = [
    (r"/Users/aayambansal/Desktop/research/research-repos/CERBench/analysis/synthetic-science-search", "<REPO_ROOT>"),
    (r"/Users/aayambansal/\.config/openscience/data-root/conda/envs/python/bin/python", "python"),
    (r"/Users/aayambansal", "<HOME>"),
    (r"https?://github\.com/aayambansal/CER-Bench(\.git)?", "<REPO_URL>"),
    (r"https?://huggingface\.co/datasets/aayambansall/CER-Bench", "<DATA_URL>"),
    (r"https?://huggingface\.co/aayambansall/[A-Za-z0-9._-]+", "<MODEL_URL>"),
    (r"aayam@syntheticsciences\.ai", "author1@anonymous.example"),
    (r"ishaan@syntheticsciences\.ai", "author2@anonymous.example"),
    (r"syntheticsciences\.ai", "anonymous.example"),
    (r"Aayam Bansal", "Anonymous Author 1"),
    (r"Ishaan Gangwani", "Anonymous Author 2"),
    (r"Bansal, Aayam and Gangwani, Ishaan", "Anonymous Authors"),
    (r"bansal2026fixedrankings", "anon2026fixedrankings"),
    (r"Synthetic Sciences", "Anonymous Institution"),
    (r"aayambansall", "anon"),
    (r"aayambansal", "anon"),
    (r"\bBansal\b", "Anon1"),
    (r"\bGangwani\b", "Anon2"),
    (r"Aayams-MacBook-Pro-3\.local", "anon-host.local"),
    (r"\bAayams?\b", "Anon"),
    (r"\bIshaan\b", "Anon2"),
]
# Case-sensitive on names so that lowercase vocabulary tokens (e.g. the surname "bansal"
# occurring as a word in the BM25 vocabulary) are not flagged; a bare literal "/Users/" in a
# blocklist is fine, a path with a username is not.
LEAK_CHECK = re.compile(r"Aayam|aayam|Bansal|Gangwani|Ishaan|ishaan|syntheticsciences|Synthetic Sciences|/Users/[A-Za-z]")

DROP = [
    "paper/release/cerbench_arxiv.tex",
    "paper/release/cerbench_arxiv.pdf",
    "paper/release/arxiv_bundle.tar.gz",
    "paper/release/OPENREVIEW_FORM.md",
    "paper/release/make_supplement.py",
]

ANON_README = """# Supplementary material (anonymized)

Code, task files, saved rankings, judgments, traces, audits, and paper source accompanying
*Fixed Rankings, Moving Leaders: A Relevance-Label and Source-Identity Audit of Scientific
Retrieval Evaluation* (double-blind review copy: `paper/release/cerbench_openreview.pdf`).

This archive is the public repository snapshot with author-identifying strings and absolute
machine paths replaced by placeholders (`<REPO_ROOT>`, `<HOME>`, `<REPO_URL>`, `<DATA_URL>`,
`<MODEL_URL>`). Provenance manifests therefore still record hashes computed under the original
paths; re-running hash-bound verification requires the public release, which will be linked
upon publication. Article text is not included anywhere; see `.gitignore` and the paper's
Ethics statement. Start with `NONHUMAN_REVISION_STATUS.md`, then
`paper/iclr2027_final/CLAIMS_TO_ARTIFACTS.md` (every number in the paper -> the file that
produced it), then the appendix N reproduction commands.
"""


def scrub(text: str) -> str:
    for pat, rep in REPLACEMENTS:
        text = re.sub(pat, rep, text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/release/supplementary_material.zip")
    args = ap.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    out = (root / args.out).resolve()
    with tempfile.TemporaryDirectory() as td:
        tree = Path(td) / "supplement"
        tree.mkdir()
        tar = subprocess.Popen(["git", "-C", str(root), "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE)
        subprocess.check_call(["tar", "-x", "-C", str(tree)], stdin=tar.stdout)
        tar.wait()
        for rel in DROP:
            p = tree / rel
            if p.exists():
                p.unlink()
        (tree / "README.md").write_text(ANON_README)
        (tree / "paper" / "release" / "README.md").write_text(
            "# Release manuscript (anonymized copy)\n\n`cerbench_openreview.pdf` is the double-blind review copy; "
            "`cerbench_body.tex` + `cerbench_openreview.tex` rebuild it with `./build.sh openreview`.\n")
        changed = 0
        leaks = []
        for p in tree.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in TEXT_EXT or p.name == ".gitignore":
                try:
                    s = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                s2 = scrub(s)
                if s2 != s:
                    p.write_text(s2, encoding="utf-8")
                    changed += 1
                if LEAK_CHECK.search(s2):
                    leaks.append(str(p.relative_to(tree)))
            elif p.suffix.lower() == ".pdf":
                try:
                    import fitz  # PyMuPDF, optional
                    txt = "".join(pg.get_text() for pg in fitz.open(str(p)))
                    if LEAK_CHECK.search(txt):
                        p.unlink()
                        print(f"dropped PDF with identifying text: {p.relative_to(tree)}")
                except ImportError:
                    pass
        if leaks:
            print("LEAK CHECK FAILED in:", *leaks, sep="\n  ")
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(tree.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(tree))
        n = sum(1 for _ in tree.rglob("*") if _.is_file())
        print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {n} files, {changed} text files scrubbed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
