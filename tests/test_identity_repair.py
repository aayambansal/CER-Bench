import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus.identity import (identity_index, normalize_id, publish_directory,
                                read_jsonl, resolve, sha256, transform_refs,
                                unique, write_jsonl)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = load_script("39_audit_repair_identity.py")
builder = load_script("04_build_corpus.py")


@pytest.mark.parametrize("value,ns,expected", [
    (" 00042 ", "PMID", "PMID:42"), ("pmid:00042", None, "PMID:42"),
    ("https://pubmed.ncbi.nlm.nih.gov/42/", None, "PMID:42"),
    ("42", "PMCID", "PMC42"), ("pmc00042", None, "PMC42"),
    ("PMCID: PMC42", None, "PMC42"),
    ("https://pmc.ncbi.nlm.nih.gov/articles/PMC42/", None, "PMC42"),
])
def test_normalization(value, ns, expected):
    assert normalize_id(value, ns) == expected


@pytest.mark.parametrize("value,ns", [("42", None), ("PMC42", "PMID"), ("PMID:42", "PMCID"),
                                         ("PMC0", None), (True, "PMID"), ("abc", "PMCID")])
def test_normalization_rejects_guessing(value, ns):
    with pytest.raises(ValueError):
        normalize_id(value, ns)


def test_ambiguity_and_cross_namespace_collision():
    rows = [{"pmid": str(p), "pmcid": c, "doc_id": c or str(p)}
            for p, c in [(1, "6286148"), (2, "PMC6286148"), (3, "6286148"), (6286148, "")]]
    index = identity_index(rows)
    assert len(index["6286148"]) == 4
    assert len(index["PMC6286148"]) == 3
    for value in ("6286148", "PMC6286148", "unknown"):
        with pytest.raises(ValueError):
            resolve(value, index)
    assert resolve("PMID:6286148", index) == "PMID:6286148"
    with pytest.raises(ValueError):
        transform_refs({"qrels": {"6286148": 2}}, lambda v: resolve(v, index))


def test_duplicate_and_qrels_alias_collapse_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        unique([{"task_id": "a"}, {"task_id": "a"}], "task_id")
    with pytest.raises(ValueError, match="collapse"):
        transform_refs({"qrels": {"1": 1, "PMID:1": 2}}, lambda v: "PMID:1")


def test_atomic_failure_and_no_overwrite(tmp_path):
    output = tmp_path / "out"
    def fail(stage):
        (stage / "partial").write_text("partial")
        raise ValueError("validation failed")
    with pytest.raises(ValueError):
        publish_directory(output, fail)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    output.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(output, lambda stage: None)
    assert list(output.iterdir()) == []
    (output / "original").write_text("preserve")
    with pytest.raises(FileExistsError):
        publish_directory(output, fail)
    assert (output / "original").read_text() == "preserve"


def test_legacy_builder_guards():
    with pytest.raises(ValueError, match="Conflicting"):
        builder.validate_build_inputs([{"pmid": "1", "pmcid": "42"}, {"pmid": "2", "pmcid": "PMC42"}], [])
    with pytest.raises(ValueError, match="duplicate"):
        builder.validate_build_inputs([{"pmid": "1"}, {"pmid": "PMID:1"}], [])
    with pytest.raises(ValueError, match="independent PMID"):
        builder.validate_build_inputs([{"pmid": "1", "pmcid": "42"}], [{"pmcid": "42"}])


def fixture_source(root):
    docs = []
    for pmid in (1, 2, 3, 4):
        docs.append({"pmid": str(pmid), "pmcid": "42" if pmid < 3 else "",
                     "doc_id": "42" if pmid < 3 else str(pmid), "title": f"Title {pmid}",
                     "abstract": f"Abstract evidence for paper {pmid}.", "has_fulltext": pmid < 3,
                     "sections": [{"text": "Wrong fulltext"}] if pmid < 3 else []})
    chunks = [{"chunk_id": f"{d['doc_id']}_abstract_0", "doc_id": d["doc_id"], "pmid": d["pmid"],
               "pmcid": d["pmcid"], "section_type": "abstract", "text": d["abstract"]} for d in docs]
    tasks = [{"task_id": f"t{p}", "question": f"Unique question {p}", "split": "wrong",
              "supporting_doc_ids": ["42" if p < 3 else str(p)], "hard_negative_doc_ids": []}
             for p in (1, 3, 4)]
    files = {"data/processed/corpus.jsonl": docs, "data/processed/chunks.jsonl": chunks,
             "data/raw/metadata/pubmed_openalex_metadata.jsonl": docs,
             "data/interim/parsed/parsed_documents.jsonl": [{"pmcid": "42", "title": "Different article"}],
             "data/benchmark/train.jsonl": tasks, "data/benchmark/dev.jsonl": [], "data/benchmark/test.jsonl": []}
    for name, rows in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, rows)
    xml = root / "data/raw/fulltext/42.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    xml.write_text('<article><front><article-meta><article-id pub-id-type="pmcid">PMC42</article-id>'
                   '<article-id pub-id-type="pmid">99</article-id><title-group><article-title>Different article</article-title>'
                   '</title-group></article-meta></front><back><article-id pub-id-type="pmid">1</article-id></back></article>')
    return files


def test_audit_conservative_candidate_and_determinism(tmp_path):
    root = tmp_path / "source"
    files = fixture_source(root)
    before = {name: sha256(root / name) for name in files}
    output = tmp_path / "out"
    report = audit_module.audit(root, output)
    assert report["release_blocked"] is True
    assert report["original"]["corpus_records"] == 4
    assert report["raw_front_pmid_disagreement_records"] == 2
    assert report["quarantine"]["chunk_records"] == 2
    assert report["quarantine"]["tasks"] == 1
    assert all(not d["has_fulltext"] and not d["sections"] and not d["pmcid"] for d in read_jsonl(output / "corpus.jsonl"))
    assert {c["doc_id"] for c in read_jsonl(output / "chunks.jsonl")} == {"PMID:3", "PMID:4"}
    assert all(t["split"] == "train" for t in read_jsonl(output / "train.jsonl"))
    evidence = read_jsonl(output / "raw_identity_evidence.jsonl")
    assert evidence[0]["pmids"] == ["PMID:99"]  # reference IDs never establish article identity
    second = audit_module.audit(root, tmp_path / "second")
    assert second == report
    assert before == {name: sha256(root / name) for name in files}
    with pytest.raises(FileExistsError):
        audit_module.audit(root, output)


def test_duplicate_task_blocks_audit_publication(tmp_path):
    fixture_source(tmp_path)
    path = tmp_path / "data/benchmark/train.jsonl"
    rows = read_jsonl(path)
    write_jsonl(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        audit_module.audit(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
