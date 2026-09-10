import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus.identity import SPLITS, read_jsonl, sha256, write_jsonl

spec = importlib.util.spec_from_file_location("splits38", ROOT / "scripts/38_make_component_disjoint_splits.py")
splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splits)


def task(tid, question=None, **extra):
    return {"task_id": tid, "question": question or f"Distinct vocabulary {tid}",
            "task_family": "family", "supporting_doc_ids": [], "hard_negative_doc_ids": [],
            "split": "stale", **extra}


def source(root, rows):
    root.mkdir()
    for split in SPLITS:
        write_jsonl(root / f"{split}.jsonl", rows if split == "train" else [])


def test_all_relations_grouped_stale_fields_and_determinism(tmp_path):
    rows = [task("a", supporting_doc_ids=["PMID:1"]), task("b", hard_negative_doc_ids=["pmid:1"]),
            task("c", supporting_passages=[{"doc_id": "PMC9"}]), task("d", qrels={"pmc9": 1}),
            task("e", "one two three four five six seven eight nine ten"),
            task("f", "one two three four five six seven eight nine ten eleven"),
            task("g", "EXACT punctuation!"), task("h", "exact punctuation"),
            task("i", evidence_cluster_id="explicit"), task("j", cluster_ids=["explicit", "transitive"]),
            task("k", evidence_cluster_id="transitive"), task("l", supporting_chunk_ids=["chunk"]),
            task("m", supporting_chunk_ids=["chunk"])]
    src = tmp_path / "src"
    source(src, rows)
    hashes = {s: sha256(src / f"{s}.jsonl") for s in SPLITS}
    report = splits.make_splits(src, tmp_path / "out", 7)
    assert report["stale_source_split_fields_corrected"] == len(rows)
    assert not any(report["cross_split_overlap"].values())
    output = {s: read_jsonl(tmp_path / "out" / f"{s}.jsonl") for s in SPLITS}
    lookup = {t["task_id"]: (s, t) for s in SPLITS for t in output[s]}
    for group in (("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"), ("i", "j", "k"), ("l", "m")):
        assert len({lookup[t][0] for t in group}) == 1
        assert len({lookup[t][1]["component_id"] for t in group}) == 1
    assert all(t["split"] == s and t["source_split"] == "train" for s in SPLITS for t in output[s])
    assert hashes == {s: sha256(src / f"{s}.jsonl") for s in SPLITS}
    assert report == splits.make_splits(src, tmp_path / "again", 7)
    with pytest.raises(FileExistsError):
        splits.make_splits(src, tmp_path / "out", 7)


@pytest.mark.parametrize("kind", ["document", "negative", "passage", "exact", "near", "cluster", "chunk", "component", "stale"])
def test_validation_rejects_each_cross_split_leak(kind):
    a, b = task("a", "alpha beta gamma delta epsilon zeta eta theta iota kappa"), task("b", "unrelated separate concepts")
    if kind == "document":
        a["supporting_doc_ids"] = b["supporting_doc_ids"] = ["same"]
    if kind == "negative":
        a["supporting_doc_ids"], b["hard_negative_doc_ids"] = ["same"], ["same"]
    if kind == "passage":
        a["supporting_passages"] = b["supporting_passages"] = [{"doc_id": "same"}]
    if kind == "exact":
        b["question"] = a["question"]
    if kind == "near":
        b["question"] = a["question"] + " lambda"
    if kind == "cluster":
        a["evidence_cluster_id"], b["cluster_ids"] = "cluster", ["cluster"]
    if kind == "chunk":
        a["chunk_ids"] = b["chunk_ids"] = ["chunk"]
    for split, t in (("train", a), ("test", b)):
        t.update(split=split, source_split="train", component_id=t["task_id"])
    if kind == "component":
        b["component_id"] = a["component_id"]
    if kind == "stale":
        b["split"] = "train"
    with pytest.raises(ValueError):
        splits.validate({"train": [a], "dev": [], "test": [b]})


def test_duplicate_task_and_failed_validation_publish_nothing(tmp_path, monkeypatch):
    src = tmp_path / "src"
    source(src, [task("a"), task("a")])
    with pytest.raises(ValueError, match="duplicate"):
        splits.make_splits(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    write_jsonl(src / "train.jsonl", [task("a")])
    original = splits.validate
    calls = []
    def fail_on_readback(rows, threshold):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("Injected readback validation failure")
        return original(rows, threshold)
    monkeypatch.setattr(splits, "validate", fail_on_readback)
    with pytest.raises(ValueError, match="Injected"):
        splits.make_splits(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))


def test_empty_source_fails_closed(tmp_path):
    src = tmp_path / "src"
    source(src, [])
    with pytest.raises(ValueError, match="No tasks"):
        splits.make_splits(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()
