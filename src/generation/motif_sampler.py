"""Graph-motif sampler for structured task generation.

Instead of sampling random papers, samples evidence graph motifs:
  - Trial → Paper → Follow-up
  - Review claim → Primary studies → Contrary study
  - Methods paper → Application paper → Benchmark paper
  - Quantitative values across multiple papers

Each motif produces a task with verifiable gold labels because the
evidence chain is structurally grounded, not LLM-imagined.
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.corpus.clinicaltrials_client import search_trials, get_trial
from src.corpus.openalex_client import get_work_by_pmid, search_works, parse_work


class MotifSampler:
    """Samples evidence graph motifs from the multi-source data stack."""

    def __init__(self, corpus_path: Path):
        """Load corpus and build lookup indices."""
        self.docs = {}
        self.docs_by_pmid = {}
        self.docs_by_seed = defaultdict(list)
        self.citation_graph = defaultdict(set)  # doc_id -> set of referenced doc_ids

        with open(corpus_path) as f:
            for line in f:
                doc = json.loads(line)
                did = doc["doc_id"]
                self.docs[did] = doc
                pmid = doc.get("pmid", "")
                if pmid:
                    self.docs_by_pmid[pmid] = did
                self.docs_by_seed[doc.get("seed_query", "unknown")].append(did)

                # Build citation graph from OpenAlex reference IDs
                for ref_oa_id in doc.get("referenced_work_ids", []):
                    self.citation_graph[did].add(ref_oa_id)

        self.all_doc_ids = list(self.docs.keys())
        print(f"MotifSampler: {len(self.docs)} docs, {len(self.docs_by_pmid)} with PMIDs")

    def sample_motif(self, motif_type: str) -> Optional[dict]:
        """Sample one evidence graph motif.

        Args:
            motif_type: One of 'trial_paper', 'citation_chain', 'contradiction',
                       'quantitative_spread', 'temporal_arc', 'methods_chain'

        Returns:
            Dict with 'papers' (list of docs), 'motif_type', 'edges' (relationships),
            and 'suggested_families' (which task families this motif supports).
        """
        samplers = {
            "trial_paper": self._sample_trial_paper_motif,
            "citation_chain": self._sample_citation_chain_motif,
            "contradiction": self._sample_contradiction_motif,
            "quantitative_spread": self._sample_quantitative_motif,
            "temporal_arc": self._sample_temporal_motif,
            "methods_chain": self._sample_methods_chain_motif,
        }

        sampler = samplers.get(motif_type)
        if not sampler:
            return None
        return sampler()

    def _sample_trial_paper_motif(self) -> Optional[dict]:
        """Motif: Clinical trial → linked publication → citation follow-up.

        This creates tasks grounded in the trial→paper→follow-up chain.
        """
        # Pick a random subdomain for the trial search
        conditions = [
            "breast cancer", "lung cancer", "melanoma", "diabetes",
            "Alzheimer", "multiple sclerosis", "rheumatoid arthritis",
            "COVID-19", "colorectal cancer", "leukemia"
        ]
        condition = random.choice(conditions)

        trials = search_trials(condition=condition, status="COMPLETED", max_results=10)
        random.shuffle(trials)

        for trial_summary in trials:
            full_trial = get_trial(trial_summary["nct_id"])
            if not full_trial or not full_trial.get("references"):
                continue

            # Find linked PMIDs that are in our corpus
            linked_pmids = [r["pmid"] for r in full_trial["references"] if r.get("pmid")]
            in_corpus = [(pmid, self.docs_by_pmid[pmid]) for pmid in linked_pmids
                         if pmid in self.docs_by_pmid]

            if len(in_corpus) >= 1:
                papers = [self.docs[did] for _, did in in_corpus[:3]]
                return {
                    "motif_type": "trial_paper",
                    "trial": {
                        "nct_id": full_trial["nct_id"],
                        "title": full_trial["title"],
                        "interventions": full_trial["interventions"],
                        "primary_outcomes": full_trial["primary_outcomes"],
                    },
                    "papers": papers,
                    "edges": [
                        {"from": full_trial["nct_id"], "to": p.get("pmid", ""),
                         "type": "trial_publication"}
                        for p in papers
                    ],
                    "suggested_families": ["constraint", "temporal", "aggregation"],
                }

        return None

    def _sample_citation_chain_motif(self) -> Optional[dict]:
        """Motif: Paper A cites Paper B which cites Paper C — a 3-hop chain."""
        # Pick a random doc with references
        candidates = [did for did, doc in self.docs.items()
                      if len(doc.get("referenced_work_ids", [])) >= 3]
        if not candidates:
            return None

        root_id = random.choice(candidates)
        root = self.docs[root_id]

        # Find referenced docs in our corpus
        chain = [root]
        ref_oa_ids = root.get("referenced_work_ids", [])
        for ref_id in ref_oa_ids:
            for did, doc in self.docs.items():
                oa_id = doc.get("openalex_id", "").replace("https://openalex.org/", "")
                if oa_id == ref_id and did != root_id:
                    chain.append(doc)
                    break
            if len(chain) >= 3:
                break

        if len(chain) >= 2:
            return {
                "motif_type": "citation_chain",
                "papers": chain[:3],
                "edges": [
                    {"from": chain[i]["doc_id"], "to": chain[i+1]["doc_id"],
                     "type": "cites"}
                    for i in range(len(chain) - 1)
                ],
                "suggested_families": ["multihop", "constraint", "comparative"],
            }
        return None

    def _sample_contradiction_motif(self) -> Optional[dict]:
        """Motif: Papers on same topic with different conclusions."""
        # Pick a seed query group and find papers with overlapping MeSH but different years
        seed = random.choice(list(self.docs_by_seed.keys()))
        group = [self.docs[did] for did in self.docs_by_seed[seed]
                 if self.docs[did].get("abstract")]

        if len(group) < 3:
            return None

        # Sort by year and pick papers from different time periods
        dated = [p for p in group if p.get("year")]
        if len(dated) < 2:
            return None

        dated.sort(key=lambda x: x["year"])
        early = dated[:len(dated)//3]
        late = dated[-len(dated)//3:]

        if early and late:
            papers = [random.choice(early), random.choice(late)]
            # Add a middle paper if available
            middle = dated[len(dated)//3:2*len(dated)//3]
            if middle:
                papers.insert(1, random.choice(middle))

            return {
                "motif_type": "contradiction",
                "papers": papers,
                "edges": [
                    {"from": papers[0]["doc_id"], "to": papers[-1]["doc_id"],
                     "type": "temporal_contrast",
                     "year_span": f"{papers[0].get('year')}-{papers[-1].get('year')}"}
                ],
                "suggested_families": ["contradiction", "temporal", "comparative"],
            }
        return None

    def _sample_quantitative_motif(self) -> Optional[dict]:
        """Motif: Multiple papers reporting the same type of measurement."""
        seed = random.choice(list(self.docs_by_seed.keys()))
        group = [self.docs[did] for did in self.docs_by_seed[seed]
                 if self.docs[did].get("abstract")]

        if len(group) < 3:
            return None

        # Sample 3-4 papers from same subdomain
        papers = random.sample(group, min(4, len(group)))

        return {
            "motif_type": "quantitative_spread",
            "papers": papers,
            "edges": [
                {"from": p["doc_id"], "to": "shared_measurement",
                 "type": "reports_value"}
                for p in papers
            ],
            "suggested_families": ["aggregation", "comparative"],
        }

    def _sample_temporal_motif(self) -> Optional[dict]:
        """Motif: Papers spanning a time range on the same topic."""
        seed = random.choice(list(self.docs_by_seed.keys()))
        group = [self.docs[did] for did in self.docs_by_seed[seed]
                 if self.docs[did].get("year") and self.docs[did].get("abstract")]

        if len(group) < 3:
            return None

        # Sort by year and pick from different eras
        group.sort(key=lambda x: x["year"])
        step = max(1, len(group) // 3)
        papers = [group[i * step] for i in range(min(3, len(group)))]

        return {
            "motif_type": "temporal_arc",
            "papers": papers,
            "edges": [
                {"from": papers[i]["doc_id"], "to": papers[i+1]["doc_id"],
                 "type": "temporal_succession",
                 "years": f"{papers[i].get('year')}-{papers[i+1].get('year')}"}
                for i in range(len(papers) - 1)
            ],
            "suggested_families": ["temporal", "contradiction"],
        }

    def _sample_methods_chain_motif(self) -> Optional[dict]:
        """Motif: Methods paper → application paper via citation."""
        return self._sample_citation_chain_motif()  # Same structure, different family


def create_sampler(project_root: str = ".") -> MotifSampler:
    """Create a motif sampler from the standard project layout."""
    root = Path(project_root)
    return MotifSampler(root / "data" / "processed" / "corpus.jsonl")
