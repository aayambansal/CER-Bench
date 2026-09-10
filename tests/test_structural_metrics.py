import copy
import pytest
from src.evaluation.structural_metrics import score_structure, validate_annotation, summarize_structure


def annotation(family="constraint"):
    prefixes = {"constraint": "constraint", "comparative": "side", "contradiction": "claim", "multihop": "hop",
                "temporal": "bin", "aggregation": "study_value", "negative_result": "negative_result"}
    prefix = prefixes[family]
    units = [prefix + ":a", prefix + ":b"]
    return {"schema_version": "cerbench.structural.v1", "task_id": "synthetic", "task_family": family,
            "annotation_status": "human_adjudicated", "required_units": units,
            "evidence_units": {"d1": [units[0]], "d2": [units[1]], "d3": [units[0]], "d4": [units[1]]},
            "valid_pairs": [["d1", "d2"], ["d3", "d4"]] if family == "contradiction" else [],
            "valid_paths": [["d1", "d2"], ["d3", "d4"]] if family == "multihop" else [],
            "provenance": {"evidence": {d: "SYNTHETIC fixture, not a real adjudication" for d in ["d1", "d2", "d3", "d4"]}}}


def test_annotation_gate_and_no_vacuous_success():
    assert score_structure([])["denominator"] == 0
    a = annotation()
    a["annotation_status"] = "unvalidated"
    assert score_structure(["d1"], a)["status"] == "not_evaluated"
    proxy = score_structure(["d1"], a, allow_proxy=True)
    assert proxy["status"] == "descriptive_proxy" and proxy["warnings"]
    assert summarize_structure([proxy])["evaluated"]["denominator"] == 0
    a["annotation_status"] = "expert_adjudicated"
    a["provenance"] = {}
    assert score_structure(["d1"], a)["denominator"] == 0
    a["required_units"] = []
    with pytest.raises(ValueError):
        score_structure([], a, allow_proxy=True)


def test_constraint_default_same_document():
    a = annotation()
    score = score_structure(["d1", "d2"], a)["metrics"]
    assert score == {"unit_coverage": 1, "set_completion": 1, "family_success": 0}
    a["constraint_policy"] = "set_level"
    assert score_structure(["d1", "d2"], a)["metrics"]["family_success"] == 1


@pytest.mark.parametrize("family", ["contradiction", "multihop"])
def test_arbitrary_union_is_not_valid_pair_or_path(family):
    a = annotation(family)
    assert score_structure(["d1", "d4"], a)["metrics"]["set_completion"] == 1
    assert score_structure(["d1", "d4"], a)["metrics"]["family_success"] == 0
    assert score_structure(["d1", "d2"], a)["metrics"]["family_success"] == 1


@pytest.mark.parametrize("family", ["comparative", "temporal", "aggregation"])
def test_explicit_unit_families(family):
    a = annotation(family)
    assert score_structure(["d1"], a)["metrics"]["unit_coverage"] == 0.5
    assert score_structure([], a)["metrics"]["family_success"] == 0
    assert score_structure(["unknown"], a)["metrics"]["family_success"] == 0
    assert score_structure(["d1", "d2"], a)["metrics"]["family_success"] == 1


def test_explicit_negative_result_hit_not_empty_gold():
    a = annotation("negative_result")
    a["required_units"] = ["negative_result:target"]
    a["evidence_units"] = {"negative": ["negative_result:target"]}
    a["provenance"] = {"evidence": {"negative": "SYNTHETIC explicit negative finding"}}
    assert score_structure([], a)["metrics"]["family_success"] == 0
    assert score_structure(["irrelevant"], a)["metrics"]["family_success"] == 0
    assert score_structure(["negative"], a)["metrics"]["family_success"] == 1


@pytest.mark.parametrize("change", [
    {"required_units": ["bogus:a", "bogus:b"]},
    {"valid_paths": [[]]}, {"valid_paths": [["d1", "d1"]]},
    {"valid_paths": [["d1", "absent"]]}, {"valid_paths": [["d1", "d3"]]},
    {"valid_paths": [["d1", "d2"], ["d1", "d2"]]},
    {"evidence_units": {"d1": []}}, {"annotation_status": "verified"},
])
def test_invalid_annotations(change):
    a = copy.deepcopy(annotation("multihop"))
    a.update(change)
    with pytest.raises(ValueError):
        validate_annotation(a)


def test_corpus_annotation_integrity():
    with pytest.raises(ValueError, match="unknown corpus"):
        validate_annotation(annotation(), ["d1"])


@pytest.mark.parametrize("pairs", [[["d1"]], [["d1", "d2", "d3"]], [["d1", "d3"]],
                                  [["d1", "d2"], ["d2", "d1"]]])
def test_malformed_pairs(pairs):
    a = annotation("contradiction")
    a["valid_pairs"] = pairs
    with pytest.raises(ValueError):
        validate_annotation(a)


def test_constraint_single_document_and_proxy_provenance():
    a = annotation()
    a["evidence_units"]["d1"] = a["required_units"][:]
    assert score_structure(["d1"], a)["metrics"]["family_success"] == 1
    a["provenance"]["evidence"]["d1"] = " "
    assert score_structure(["d1"], a)["status"] == "not_evaluated"
    assert score_structure(["d1"], a, True)["status"] == "descriptive_proxy"
