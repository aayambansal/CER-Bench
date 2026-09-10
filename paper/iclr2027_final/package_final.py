"""Package only paper sources; omit raw evidence, credentials and coordinator data."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["cerbench_final.tex", "refs.bib", "iclr2027_conference.sty",
         "iclr2027_conference.bst", "natbib.sty", "fancyhdr.sty",
         "figures/qrel_disclosure_readable.pdf", "README.md", "CLAIMS_TO_ARTIFACTS.md",
         "pdf_checks.json"]
checks = json.loads((HERE / "pdf_checks.json").read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert checks["pdf_sha256"] == sha(HERE / "cerbench_final.pdf")
assert checks["tex_sha256"] == sha(HERE / "cerbench_final.tex")
assert checks["within_nine_page_main_limit"] and checks["all_fonts_embedded"]
assert not checks["latex_issues"] and not checks["private_text_matches"]
for relative in FILES:
    p = HERE / relative
    if p.suffix in {".md", ".tex", ".bib", ".json", ".py"}:
        text = p.read_text()
        assert "/Users/" not in text and "aayambansal" not in text.lower()
        assert not re.search(r"sk-(?:proj|ant)-[A-Za-z0-9_-]{20,}", text)
        assert not re.search(r"AIza[A-Za-z0-9_-]{30,}", text)
out = HERE / "CERBench_ICLR2027_final_source.zip"
with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for relative in FILES:
        info = zipfile.ZipInfo("CERBench_ICLR2027_final/" + relative, date_time=(2026, 9, 9, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, (HERE / relative).read_bytes())
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for relative in FILES:
        assert z.read("CERBench_ICLR2027_final/" + relative) == (HERE / relative).read_bytes()
manifest = {"pdf": {"path": "cerbench_final.pdf", "sha256": sha(HERE / "cerbench_final.pdf")},
            "source_archive": {"path": out.name, "sha256": sha(out), "files": FILES},
            "main_text_pages": checks["main_text_end_page"], "total_pages": checks["total_pages"],
            "complete_manuscript_delivered": True, "acceptance_guaranteed": False,
            "human_relevance_validation": False, "new_model_experiments": 0,
            "scope": "source_identity_and_historical_label_audit_not_corrected_benchmark_leaderboard"}
(HERE / "FINAL_DELIVERABLES.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
