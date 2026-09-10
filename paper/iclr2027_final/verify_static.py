"""Provider-free source checks; deliberately not a TeX compiler or result rerun."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tex = (HERE / "cerbench_final.tex").read_text()
    bib = (HERE / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
    citations = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
        citations.update(k.strip() for k in group.split(","))
    assert 10 <= len(citations) <= 15, citations
    assert not citations - keys, citations - keys
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    assert not set(refs) - set(labels), set(refs) - set(labels)
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", tex):
        if kind == "begin":
            stack.append(env)
        else:
            assert stack and stack.pop() == env, (kind, env)
    assert not stack, stack
    clean = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", tex, flags=re.S)
    clean = re.sub(r"(?<!\\)%[^\n]*", "", clean)
    depth = 0
    for char in re.sub(r"\\[{}]", "", clean):
        depth += (char == "{") - (char == "}")
        assert depth >= 0
    assert depth == 0, depth
    for forbidden in ("/Users/", "/private/", "TODO", "TBD", "PLACEHOLDER", "internal working draft", "\\vspace", "\\geometry", "\\setlength", "\\iclrfinalcopy"):
        assert forbidden not in tex, forbidden
    files = {}
    for name in ("refs.bib", "iclr2027_conference.sty", "iclr2027_conference.bst", "natbib.sty", "fancyhdr.sty"):
        assert digest(HERE / name) == digest(ROOT / "paper/iclr2027_submission" / name)
        files[name] = digest(HERE / name)
    fig = HERE / "figures/qrel_disclosure.pdf"
    assert digest(fig) == digest(ROOT / "results/readiness/qrel_disclosure/v2/qrel_disclosure.pdf")
    assert digest(fig) == "155ef70e796a739f58f9963a706f1c42ea12a78a125f7861daf1e0803b4bff09"
    files["figures/qrel_disclosure.pdf"] = digest(fig)
    a = json.loads((ROOT / "results/readiness/authoritative_corpus/summary.json").read_text())
    assert a["parsed_rows"] == 4936
    assert a["metadata_changed_rows_excluding_doc_id"] == 815
    assert a["change_counts"]["pmcid"] == 247
    assert a["changed_old_ids_found_in_current_references"]["pmcid"] == 226
    d = json.loads((ROOT / "results/readiness/qrel_disclosure/v2/summary.json").read_text())
    assert d["denominators"] == dict(empty_excluded=17, full_pairs=1370, raw_judgments=3240, seed_pairs=264, supported=108)
    fractions = {x["fraction"]: x for x in d["fractions"]}
    for f, rate in ((.2, .188), (.3, .567), (.5, .92), (1., 1.)):
        assert fractions[f]["systems"]["bge"]["top1_tie_adjusted_win_rate"] == rate
    for row in d["bge_minus_agent"]:
        if row["fraction"] in (.2, .3):
            lo, hi = row["difference"]["empirical95"]
            assert lo < 0 < hi
    figure_names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    assert all((HERE / name).is_file() for name in figure_names)
    report = {
        "status": "passed",
        "scope": "static source checks and selected evidence assertions only",
        "citations": sorted(citations),
        "cited_entry_count": len(citations),
        "bibliography_entry_count": len(keys),
        "cross_references_resolved": len(refs),
        "environments_and_braces_balanced": True,
        "unchanged_copy_sha256": files,
        "manuscript_sha256": digest(HERE / "cerbench_final.tex"),
        "main_raw_whitespace_tokens_before_statements": len(tex.split("\\section*{Reproducibility Statement}")[0].split()),
        "tex_compiled": False,
        "pdf_pagination_or_visual_checks": False,
        "empirical_suites_rerun": False,
        "new_model_calls": 0,
    }
    (HERE / "static_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
