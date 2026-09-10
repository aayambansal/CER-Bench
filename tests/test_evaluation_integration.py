from src.evaluation.structural_metrics import score_structure


def test_actual_dataset_negative_family_requires_explicit_evidence():
    a = {"schema_version": "cerbench.structural.v1", "task_id": "SYNTHETIC-negative",
         "task_family": "negative", "annotation_status": "unvalidated",
         "required_units": ["negative_result:null_endpoint"],
         "evidence_units": {"SYNTHETIC-doc": ["negative_result:null_endpoint"]},
         "valid_pairs": [], "valid_paths": [], "provenance": {"synthetic": True}}
    assert score_structure(["SYNTHETIC-doc"], a)["status"] == "not_evaluated"
    result = score_structure(["SYNTHETIC-doc"], a, allow_proxy=True)
    assert result["status"] == "descriptive_proxy"
    assert result["metrics"]["family_success"] == 1
    assert score_structure([], a, allow_proxy=True)["metrics"]["family_success"] == 0
