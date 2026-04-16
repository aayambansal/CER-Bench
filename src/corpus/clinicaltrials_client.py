"""ClinicalTrials.gov API v2 client for structured trial data.

Provides structured supervision: interventions, outcomes, eligibility,
results, timelines. Essential for trial→paper→follow-up graph motifs.

API: https://clinicaltrials.gov/data-api/api
"""

import time
from typing import Optional

import requests

_SESSION = requests.Session()
_BASE_URL = "https://clinicaltrials.gov/api/v2"
_RATE_LIMIT = 5
_MIN_INTERVAL = 1.0 / _RATE_LIMIT
_last_request = 0.0


def _rate_limit():
    global _last_request
    now = time.time()
    elapsed = now - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request = time.time()


def search_trials(
    condition: str = None,
    intervention: str = None,
    status: str = None,
    phase: str = None,
    max_results: int = 100,
) -> list[dict]:
    """Search ClinicalTrials.gov for trials matching criteria.

    Args:
        condition: Disease or condition (e.g., 'breast cancer')
        intervention: Drug or intervention (e.g., 'pembrolizumab')
        status: Trial status filter (e.g., 'COMPLETED')
        phase: Phase filter (e.g., 'PHASE3')
        max_results: Maximum number of results

    Returns:
        List of trial summary dicts.
    """
    _rate_limit()
    params = {
        "format": "json",
        "pageSize": min(max_results, 100),
    }

    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    if status:
        params["filter.overallStatus"] = status
    if phase:
        params["filter.phase"] = phase

    try:
        resp = _SESSION.get(f"{_BASE_URL}/studies", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies", [])
        return [_parse_trial_summary(s) for s in studies]
    except Exception as e:
        print(f"  Warning: ClinicalTrials search failed: {e}")
        return []


def get_trial(nct_id: str) -> Optional[dict]:
    """Fetch a single trial by NCT ID.

    Args:
        nct_id: NCT identifier (e.g., 'NCT04379518')

    Returns:
        Parsed trial dict or None.
    """
    _rate_limit()
    try:
        resp = _SESSION.get(f"{_BASE_URL}/studies/{nct_id}", params={"format": "json"}, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _parse_trial_full(resp.json())
    except Exception as e:
        print(f"  Warning: ClinicalTrials fetch failed for {nct_id}: {e}")
        return None


def _parse_trial_summary(study: dict) -> dict:
    """Parse a study object into a summary."""
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    desc_mod = proto.get("descriptionModule", {})

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("briefTitle", ""),
        "official_title": id_mod.get("officialTitle", ""),
        "status": status_mod.get("overallStatus", ""),
        "phase": design_mod.get("phases", []),
        "brief_summary": desc_mod.get("briefSummary", ""),
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
    }


def _parse_trial_full(study: dict) -> dict:
    """Parse a full study object with interventions, outcomes, etc."""
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    desc_mod = proto.get("descriptionModule", {})
    design_mod = proto.get("designModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    eligibility_mod = proto.get("eligibilityModule", {})
    refs_mod = proto.get("referencesModule", {})
    results_sec = study.get("resultsSection", {})

    # Interventions
    interventions = []
    for arm in arms_mod.get("interventions", []):
        interventions.append({
            "type": arm.get("type", ""),
            "name": arm.get("name", ""),
            "description": arm.get("description", "")[:500],
        })

    # Outcomes
    primary_outcomes = []
    for outcome in outcomes_mod.get("primaryOutcomes", []):
        primary_outcomes.append({
            "measure": outcome.get("measure", ""),
            "timeframe": outcome.get("timeFrame", ""),
            "description": outcome.get("description", "")[:300],
        })

    secondary_outcomes = []
    for outcome in outcomes_mod.get("secondaryOutcomes", []):
        secondary_outcomes.append({
            "measure": outcome.get("measure", ""),
            "timeframe": outcome.get("timeFrame", ""),
        })

    # References (linked publications)
    references = []
    for ref in refs_mod.get("references", []):
        references.append({
            "pmid": ref.get("pmid", ""),
            "type": ref.get("type", ""),
            "citation": ref.get("citation", "")[:300],
        })

    # Eligibility
    eligibility = {
        "criteria": eligibility_mod.get("eligibilityCriteria", "")[:1000],
        "sex": eligibility_mod.get("sex", ""),
        "min_age": eligibility_mod.get("minimumAge", ""),
        "max_age": eligibility_mod.get("maximumAge", ""),
        "healthy_volunteers": eligibility_mod.get("healthyVolunteers", ""),
    }

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("briefTitle", ""),
        "official_title": id_mod.get("officialTitle", ""),
        "status": status_mod.get("overallStatus", ""),
        "phase": design_mod.get("phases", []),
        "study_type": design_mod.get("studyType", ""),
        "brief_summary": desc_mod.get("briefSummary", ""),
        "detailed_description": desc_mod.get("detailedDescription", "")[:2000],
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
        "interventions": interventions,
        "primary_outcomes": primary_outcomes,
        "secondary_outcomes": secondary_outcomes,
        "eligibility": eligibility,
        "references": references,
        "has_results": bool(results_sec),
        "source": "clinicaltrials",
    }
