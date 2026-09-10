"""Compile a clean extraction of the delivered source ZIP and compare its PDF."""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--compiler", type=Path, required=True)
parser.add_argument("--scratch", type=Path, required=True)
args = parser.parse_args()
stage = args.scratch.resolve()
stage.mkdir(parents=True, exist_ok=False)
archive = HERE / "CERBench_ICLR2027_final_source.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        p = (stage / name).resolve()
        assert p.is_relative_to(stage) and not name.startswith("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(z.read(name))
root = stage / "CERBench_ICLR2027_final"
cmd = [str(args.compiler.resolve()), "--keep-logs", "--keep-intermediates", "cerbench_final.tex"]
r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=120)
(HERE / "source_archive_build.log").write_text(r.stdout + "\n" + r.stderr)
r.check_returncode()
reference = fitz.open(HERE / "cerbench_final.pdf")
rebuilt = fitz.open(root / "cerbench_final.pdf")
assert len(reference) == len(rebuilt)
for a, b in zip(reference, rebuilt):
    assert a.rect == b.rect and a.get_text() == b.get_text()
log = (root / "cerbench_final.log").read_text()
assert not re.search(r"Overfull \\[hv]box|undefined citations|undefined references|Citation .* undefined|Reference .* undefined", log)
result = {"status": "passed", "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
          "clean_extraction_compiles": True, "identical_extracted_text_and_page_geometry": True,
          "pages": len(rebuilt), "recompiled_pdf_sha256": hashlib.sha256((root / "cerbench_final.pdf").read_bytes()).hexdigest(),
          "reference_pdf_sha256": hashlib.sha256((HERE / "cerbench_final.pdf").read_bytes()).hexdigest(),
          "note": "PDF byte hashes may differ due to build metadata; this checks text and page geometry, not scientific validity."}
(HERE / "source_archive_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
