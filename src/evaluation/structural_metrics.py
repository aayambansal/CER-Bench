"""Annotation-gated structural evidence metrics (never derive roles from qrels)."""
from .strict_metrics import strings

ANNOTATION_VERSION = "cerbench.structural.v1"
VERIFIED = {"human_adjudicated", "expert_adjudicated"}
FAMILIES = {"constraint", "comparative", "contradiction", "multihop", "temporal", "aggregation", "negative", "negative_result"}


def has_provenance(value):
    """Evidence provenance maps each annotated document to a nonblank source string."""
    return isinstance(value, dict) and bool(value) and all(isinstance(v, str) and bool(v.strip()) for v in value.values())


def validate_annotation(a, corpus_ids=None):
    if not isinstance(a, dict) or a.get("schema_version") != ANNOTATION_VERSION:
        raise ValueError("unsupported structural annotation schema_version")
    if not isinstance(a.get("task_id"), str) or not a["task_id"].strip():
        raise ValueError("invalid annotation task_id")
    family = a.get("task_family")
    if family not in FAMILIES:
        raise ValueError(f"unsupported task_family {family}")
    if a.get("annotation_status") not in VERIFIED | {"unvalidated"}:
        raise ValueError("invalid annotation_status")
    required = set(strings(a.get("required_units"), "required_units", True))
    evidence = a.get("evidence_units")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("evidence_units must be a nonempty document-to-units map")
    strings(list(evidence), "evidence document IDs", True)
    for doc, units in evidence.items():
        if not set(strings(units, f"evidence_units[{doc}]", True)) <= required:
            raise ValueError("evidence unit not in required_units")
    if set().union(*(set(u) for u in evidence.values())) != required:
        raise ValueError("required unit has no annotated evidence")
    if corpus_ids is not None and not set(evidence) <= set(corpus_ids):
        raise ValueError("annotation references unknown corpus document")
    policy = a.get("constraint_policy", "same_document")
    if policy not in ("same_document", "set_level") or (family != "constraint" and "constraint_policy" in a):
        raise ValueError("invalid constraint_policy")
    for field, target in (("valid_pairs", "contradiction"), ("valid_paths", "multihop")):
        groups = a.get(field)
        if not isinstance(groups, list):
            raise ValueError(f"{field} must be a list")
        if family == target and not groups:
            raise ValueError(f"{family} requires nonempty {field}")
        if family != target and groups:
            raise ValueError(f"{field} not applicable to {family}")
        seen = set()
        for group in groups:
            docs = strings(group, field, True)
            if len(docs) < 2 or (field == "valid_pairs" and len(docs) != 2):
                raise ValueError(f"invalid {field} length")
            if not set(docs) <= set(evidence):
                raise ValueError(f"{field} references unannotated document")
            signature = tuple(sorted(docs)) if field == "valid_pairs" else tuple(docs)
            if signature in seen:
                raise ValueError(f"duplicate {field}")
            seen.add(signature)
            if set().union(*(set(evidence[d]) for d in docs)) != required:
                raise ValueError(f"{field} does not cover required roles/units")
            if field == "valid_pairs" and (any(len(evidence[d]) != 1 for d in docs) or evidence[docs[0]] == evidence[docs[1]]):
                raise ValueError("valid_pairs require distinct, explicit opposing claim roles")
    # Explicit unit namespaces carry roles, not task-family or seed-gold heuristics.
    prefixes = {"constraint": "constraint:", "comparative": "side:", "contradiction": "claim:",
                "multihop": "hop:", "temporal": "bin:", "aggregation": "study_value:", "negative": "negative_result:", "negative_result": "negative_result:"}
    prefix = prefixes[family]
    if any(not u.startswith(prefix) or not u[len(prefix):].strip() for u in required):
        raise ValueError(f"{family} units must use nonempty {prefix} names")
    if family in {"comparative", "contradiction"} and len(required) != 2:
        raise ValueError(f"{family} requires exactly two explicit roles")
    if family == "multihop" and len(required) < 2:
        raise ValueError("multihop requires at least two hops")
    if family in {"negative", "negative_result"} and len(required) != 1:
        raise ValueError("negative_result requires exactly one explicitly annotated target unit")
    provenance = a.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    ep = provenance.get("evidence")
    verified = a["annotation_status"] in VERIFIED and has_provenance(ep) and set(ep) == set(evidence)
    return verified


def score_structure(retrieved, annotation=None, allow_proxy=False, corpus_ids=None):
    strings(retrieved, "retrieved_doc_ids")
    if corpus_ids is not None:
        corpus = set(strings(corpus_ids, "corpus IDs"))
        if not set(retrieved) <= corpus:
            raise ValueError("retrieval references unknown corpus document")
    if annotation is None:
        return {"status": "not_evaluated", "reason": "missing_annotation", "denominator": 0, "metrics": None}
    verified = validate_annotation(annotation, corpus_ids)
    if not verified and not allow_proxy:
        return {"status": "not_evaluated", "reason": "unverified_annotation_or_missing_evidence_provenance", "denominator": 0, "metrics": None}
    a = annotation
    required = set(a["required_units"])
    docs = set(retrieved)
    observed = set().union(*(set(units) for doc, units in a["evidence_units"].items() if doc in docs))
    complete = required <= observed
    family = a["task_family"]
    success = complete
    if family == "constraint" and a.get("constraint_policy", "same_document") == "same_document":
        success = any(required <= set(units) for doc, units in a["evidence_units"].items() if doc in docs)
    elif family == "contradiction":
        success = any(set(pair) <= docs for pair in a["valid_pairs"])
    elif family == "multihop":
        success = any(set(path) <= docs for path in a["valid_paths"])
    return {"status": "evaluated" if verified else "descriptive_proxy", "denominator": 1,
            "warnings": [] if verified else ["descriptive_proxy: not verified structural evidence"],
            "metrics": {"unit_coverage": len(observed & required) / len(required),
                        "set_completion": int(complete), "family_success": int(success)}}


def summarize_structure(records):
    result = {}
    # Never pool proxy annotations into verified estimates.
    for status in ("evaluated", "descriptive_proxy"):
        eligible = [r for r in records if r["status"] == status]
        result[status] = {"denominator": len(eligible), "metrics": {
            k: sum(r["metrics"][k] for r in eligible) / len(eligible) if eligible else None
            for k in ("unit_coverage", "set_completion", "family_success")}}
    result["not_evaluated"] = sum(r["status"] == "not_evaluated" for r in records)
    return result
