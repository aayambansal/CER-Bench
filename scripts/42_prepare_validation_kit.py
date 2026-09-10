#!/usr/bin/env python3
"""Prepare PROVISIONAL, blank local review material. Standard library; no retrieval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260909
FAMILIES = ('constraint', 'comparative', 'contradiction', 'multihop',
            'temporal', 'aggregation', 'negative')
SYSTEMS = ('bm25', 'dense', 'bge', 'e5large', 'medcpt', 'splade', 'rm3',
           'hybrid', 'hybrid_reranked', 'bge_reranker', 'agent_single_step',
           'agent', 'agent_gpt54', 'colbert', 'rocchio')
EDITABLE = ('relevance', 'evidence_span', 'section_id', 'start', 'end',
            'role_json', 'independent_sufficiency', 'confidence', 'notes',
            'annotator_id', 'annotation_time', 'provenance_kind')
ROLES = ('constraint', 'comparison_A', 'comparison_B', 'finding_A', 'finding_B',
         'reconciliation', 'hop', 'temporal', 'measurement', 'explicit_null',
         'negative_direction', 'failed_replication', 'context', 'other')


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sheet(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def norm(value):
    value = unicodedata.normalize('NFKC', str(value)).strip().upper()
    if re.fullmatch(r'[0-9]+', value):
        return str(int(value))
    if re.fullmatch(r'PMC[0-9]+', value):
        return 'PMC' + str(int(value[3:]))
    return value


def pmc(value):
    value = norm(value)
    if re.fullmatch(r'(PMC)?[0-9]+', value):
        return 'PMC' + str(int(value.removeprefix('PMC')))
    return ''


def aliases(doc):
    """Bare record IDs remain lookup aliases, never a guessed URL namespace."""
    out = {norm(doc['doc_id'])}
    if doc.get('pmid'):
        out.update((norm(doc['pmid']), 'PMID:' + norm(doc['pmid'])))
    if doc.get('pmcid'):
        out.update((norm(doc['pmcid']), pmc(doc['pmcid'])))
    if doc.get('doi'):
        out.add('DOI:' + norm(doc['doi']).removeprefix('HTTPS://DOI.ORG/'))
    return out - {''}


def identity_audit(docs):
    owners = defaultdict(set)
    for i, d in enumerate(docs):
        for key in aliases(d):
            owners[key].add(i)
    conflicts = {key: sorted(ids) for key, ids in owners.items() if len(ids) > 1}
    bad = {i for ids in conflicts.values() for i in ids}
    # A conflicting record's every alias is tainted, not just its collided key.
    unsafe = {key for key, ids in owners.items() if ids & bad}
    lookup = {key: next(iter(ids)) for key, ids in owners.items() if key not in unsafe}
    return lookup, bad, conflicts


def source_urls(doc):
    urls = []
    if re.fullmatch(r'[0-9]+', str(doc.get('pmid', ''))):
        urls.append('https://pubmed.ncbi.nlm.nih.gov/' + str(doc['pmid']) + '/')
    if pmc(doc.get('pmcid', '')):
        urls.append('https://pmc.ncbi.nlm.nih.gov/articles/' + pmc(doc['pmcid']) + '/')
    return urls


def evidence(doc):
    sections = []
    def add(sid, heading, text):
        if text:
            sections.append({'section_id': sid, 'heading': heading, 'text': text})
    add('abstract', 'Abstract', doc.get('abstract', ''))
    for i, s in enumerate(doc.get('sections', [])):
        add(f'sections/{i}', s.get('heading', ''), s.get('text', ''))
    for i, s in enumerate(doc.get('figure_captions', [])):
        add(f'figure_captions/{i}', s.get('label', ''), s.get('caption', ''))
    for i, s in enumerate(doc.get('table_texts', [])):
        add(f'table_texts/{i}', s.get('label', ''), '\n'.join(
            str(s[k]) for k in ('caption', 'body_text') if s.get(k)))
    return sections


def build(root):
    out = root / 'annotations/human_qrel_v2'
    queries_out = root / 'annotations/human_queries_v1'
    if out.exists() or queries_out.exists():
        raise ValueError('Refusing to overwrite any existing v2 kit or query collection')
    inputs = {}
    def track(p):
        inputs[str(p.relative_to(root))] = sha(p)
        return p
    docs = read_jsonl(track(root / 'data/processed/corpus.jsonl'))
    tasks = read_jsonl(track(root / 'data/benchmark/test.jsonl'))
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task_id in test split')
    lookup, bad, conflicts = identity_audit(docs)
    reasons = defaultdict(list)
    for i in bad:
        reasons[i].append('normalized_identity_collision_or_alias')
    # Verify locally available raw XML identifiers before supplying merged body text.
    # No XML is fetched. Bare PMCID metadata is normalized only within that field.
    for i, d in enumerate(docs):
        if d.get('pmcid') and not pmc(d['pmcid']):
            reasons[i].append('malformed_pmcid')
        if not re.fullmatch(r'[0-9]+', str(d.get('pmid', ''))):
            reasons[i].append('missing_or_malformed_pmid')
        if norm(d['doc_id']) not in {norm(d.get('pmid', '')), norm(d.get('pmcid', '')), pmc(d.get('pmcid', ''))}:
            reasons[i].append('record_id_metadata_mismatch')
        if d.get('sections') or d.get('figure_captions') or d.get('table_texts'):
            xml = root / 'data/raw/fulltext' / (pmc(d.get('pmcid', '')) + '.xml')
            if not xml.exists():
                reasons[i].append('body_without_verifiable_local_xml_identity')
                continue
            try:
                article = ET.parse(track(xml)).getroot()
                if article.tag != 'article':
                    article = article.find('./article')
                if article is None:
                    reasons[i].append('missing_xml_article')
                    continue
                ids = defaultdict(set)
                for node in article.findall('./front/article-meta/article-id'):
                    ids[node.get('pub-id-type')].add(''.join(node.itertext()).strip())
                xml_pmcs = {pmc(x) for k in ('pmc', 'pmcid') for x in ids[k]}
                if ids['pmid'] != {str(d.get('pmid', ''))} or xml_pmcs != {pmc(d.get('pmcid', ''))}:
                    reasons[i].append('raw_xml_identity_mismatch')
            except ET.ParseError:
                reasons[i].append('malformed_local_xml')
        if not evidence(d):
            reasons[i].append('no_local_evidence')
    bad.update(reasons)
    lookup = {k: i for k, i in lookup.items() if i not in bad}
    # Selection is predeclared and made BEFORE identity exclusions; never refill.
    selected = []
    for family in FAMILIES:
        eligible = sorted((t for t in tasks if t['task_family'] == family and t.get('supporting_doc_ids')), key=lambda t: t['task_id'])
        if len(eligible) < 14:
            raise ValueError(f'Cannot predeclare 14 supported queries for {family}')
        selected.extend(random.Random(f'{SEED}:select:{family}').sample(eligible, 14))
    selected += sorted((t for t in tasks if t['task_family'] == 'abstention'), key=lambda t: t['task_id'])
    runs = {}
    missing = []
    for system in SYSTEMS:
        p = root / 'results/baselines' / f'{system}_test.jsonl'
        if not p.exists():
            missing.append(system)
            continue
        rows = read_jsonl(track(p))
        if len({r['task_id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate run task IDs: {p}')
        runs[system] = {r['task_id']: r for r in rows}
    if not runs:
        raise ValueError('No local major-system runs')
    retained, exclusions, pair_exclusions, rows, private = [], [], [], [], []
    coverage_gaps = []
    safe_ids = sorted(set(lookup.values()))
    used = set()
    for t in selected:
        tid = t['task_id']
        seeds = t.get('supporting_doc_ids', [])
        invalid_seeds = [d for d in seeds if norm(d) not in lookup]
        if invalid_seeds:
            exclusions.append({'task_id': tid, 'task_family': t['task_family'], 'reason': 'unsafe_or_missing_seed_identity', 'doc_ids': invalid_seeds})
            continue
        pool = defaultdict(set)
        for d in seeds:
            pool[str(d)].add('seed')
        for d in t.get('hard_negative_doc_ids', []):
            pool[str(d)].add('original_candidate')
        for system, run in runs.items():
            if tid not in run:
                coverage_gaps.append({'task_id': tid, 'system': system})
            for d in run.get(tid, {}).get('retrieved_docs', [])[:20]:
                pool[str(d)].add(system + ':top20')
        safe_pool = defaultdict(set)
        for d, sources in pool.items():
            if norm(d) not in lookup:
                pair_exclusions.append({'task_id': tid, 'doc_id': d, 'reason': 'unsafe_or_missing_identity', 'sources': sorted(sources)})
            else:
                safe_pool[lookup[norm(d)]].update(sources)
        tail = random.Random(f'{SEED}:tail:{tid}').sample(
            [i for i in safe_ids if i not in safe_pool], min(5, len(set(safe_ids) - set(safe_pool))))
        for i in tail:
            safe_pool[i].add('uniform_random_out_of_pool')
        retained.append(t)
        for i in sorted(safe_pool):
            d = docs[i]
            did = str(d['doc_id'])
            pair = hashlib.sha256(f'{tid}\0{did}'.encode()).hexdigest()[:24]
            ep = 'evidence/' + hashlib.sha256(did.encode()).hexdigest()[:24] + '.json'
            rows.append({'pair_id': pair, 'task_id': tid, 'task_family': t['task_family'],
                         'question': t['question'], 'required_constraints': json.dumps(t.get('required_constraints', []), ensure_ascii=False),
                         'doc_id': did, 'title': d.get('title', ''), 'year': str(d.get('year') or ''),
                         'source_urls': json.dumps(source_urls(d)), 'evidence_file': ep,
                         'text_availability': 'parsed_fulltext' if d.get('sections') else 'abstract_only',
                         **{key: '' for key in EDITABLE}})
            private.append({'pair_id': pair, 'task_id': tid, 'doc_id': did, 'sources': sorted(safe_pool[i])})
            used.add(i)
    if not rows:
        raise ValueError('All pairs quarantined; no kit can be prepared')
    out.mkdir(parents=True)
    for i in sorted(used):
        d = docs[i]
        did = str(d['doc_id'])
        ep = 'evidence/' + hashlib.sha256(did.encode()).hexdigest()[:24] + '.json'
        dump(out / ep, {'doc_id': did, 'title': d.get('title', ''), 'source_urls': source_urls(d),
                        'provenance': {'source': 'data/processed/corpus.jsonl', 'source_sha256': inputs['data/processed/corpus.jsonl'], 'line_1based': i + 1},
                        'offset_convention': 'zero-based Unicode codepoints, end exclusive, within section text',
                        'sections': evidence(d)})
    for who in ('A', 'B'):
        shuffled = rows.copy()
        random.Random(f'{SEED}:annotator:{who}').shuffle(shuffled)
        sheet(out / f'annotator_{who}.csv', shuffled)
        task_rows = [{'task_id': t['task_id'], 'task_family': t['task_family'], 'question': t['question'],
                      'constraint_requirements': '', 'required_units': '', 'valid_pair_requirements': '',
                      'valid_path_requirements': '', 'joint_sufficiency': '', 'abstention_assessment': '',
                      'scope_limitations': '', 'annotator_id': '', 'annotation_time': '', 'provenance_kind': ''} for t in retained]
        random.Random(f'{SEED}:tasks:{who}').shuffle(task_rows)
        sheet(out / f'task_requirements_{who}.csv', task_rows)
    structures = [{'task_id': t['task_id'], 'task_family': t['task_family'], 'annotation_status': 'unvalidated',
                   'required_units': [], 'evidence_units': {}, 'valid_pairs': [], 'valid_paths': [], 'provenance': {}} for t in retained]
    dump(out / 'structural_template.json', structures)
    dump(out / 'coordinator/pool_provenance.json', private)
    dump(out / 'coordinator/selection.json', [{'task_id': t['task_id'], 'task_family': t['task_family']} for t in selected])
    dump(out / 'coordinator/identity_audit.json', {
        'collisions': conflicts, 'index_convention': 'zero-based corpus record indices',
        'quarantined_records': [{'index': i, 'doc_id': docs[i]['doc_id'], 'pmid': docs[i].get('pmid'), 'pmcid': docs[i].get('pmcid'), 'reasons': reasons[i]} for i in sorted(bad)],
        'excluded_queries': exclusions, 'excluded_candidate_references': pair_exclusions})
    # Eighty blank human-written-query slots: target strata, NOT human observations.
    query_rows = []
    families = FAMILIES + ('abstention',)
    for i in range(80):
        query_rows.append({'collection_id': f'HQ{i+1:03}', 'target_family': families[i % 8],
                           'target_role': ('researcher', 'clinician', 'information_specialist', 'graduate_researcher')[(i // 8) % 4],
                           'target_domain': ('biomedicine', 'computational_biology')[(i // 16) % 2],
                           'target_fulltext_stratum': ('fulltext_available', 'abstract_only')[(i // 8) % 2],
                           **{k: '' for k in ('question', 'actual_role', 'actual_family', 'actual_domain', 'actual_fulltext_stratum', 'writer_id', 'written_at', 'independence_attestation', 'consent_record', 'compensation_terms', 'ethics_review_applicability', 'ethics_review_reference', 'notes')}})
    sheet(queries_out / 'human_queries_blank.csv', query_rows)
    dump(queries_out / 'manifest.json', {'status': 'blank_template_no_human_data', 'target_queries': 80, 'acceptable_target_range': [50, 100], 'seed_qrels_and_models_visible_before_writing': False,
                                        'output_sha256': {'human_queries_blank.csv': sha(queries_out / 'human_queries_blank.csv')}})
    counts = Counter(r['task_family'] for r in rows)
    n = len(rows)
    manifest = {'status': 'PROVISIONAL_UNVALIDATED', 'release_blocked': True,
                'blockers': ['corpus identity repair and reindexing required', 'component-disjoint split must be fixed and frozen', 'existing test set explored; not a fresh confirmatory holdout', 'no human annotators or adjudicators; templates only'],
                'seed': SEED, 'selection_rule': '14 per supported family before filtering; all abstention; no replacement',
                'supported_selected': 98, 'abstention_selected': sum(t['task_family'] == 'abstention' for t in selected),
                'retained_queries_by_family': dict(Counter(t['task_family'] for t in retained)),
                'excluded_queries': len(exclusions), 'excluded_candidate_references': len(pair_exclusions),
                'corpus_records': len(docs), 'collision_keys': len(conflicts), 'quarantined_records': len(bad),
                'pairs_by_family': dict(counts), 'pairs_total': n, 'unique_evidence_documents': len(used),
                'random_out_of_pool_pairs': sum('uniform_random_out_of_pool' in p['sources'] for p in private),
                'seed_pairs': sum('seed' in p['sources'] for p in private),
                'pairs_by_text_availability': dict(Counter(r['text_availability'] for r in rows)),
                'systems': list(runs), 'missing_systems': missing, 'run_query_coverage_gaps': coverage_gaps,
                'workload': {'pair_judgments_two_annotators': 2*n, 'task_reviews_two_annotators': 2*len(retained),
                             'pair_review_minutes_assumption': [2, 5], 'pair_review_person_hours': [round(2*n*2/60, 2), round(2*n*5/60, 2)],
                             'adjudication_10_to_30_percent_at_5_to_10_minutes_hours': [round(n*.1*5/60, 2), round(n*.3*10/60, 2)],
                             'note': 'planning assumptions, not measured effort; task review/training/writing additional'},
                'input_sha256': inputs,
                'output_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    dump(out / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('input_sha256', 'output_sha256')}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    build(parser.parse_args().root)
