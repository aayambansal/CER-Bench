#!/usr/bin/env python3
"""Fail-closed local validation. Never emits final human qrels or fills judgments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

spec = importlib.util.spec_from_file_location('kit42', Path(__file__).with_name('42_prepare_validation_kit.py'))
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)
VALID = {'0', '1', '2', 'U'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'Missing/duplicate CSV headers')
        rows = list(reader)
    require(rows, 'Empty annotation sheet')
    result = {}
    for r in rows:
        require(None not in r and all(v is not None for v in r.values()), 'Malformed CSV row')
        p = r.get('pair_id', '')
        require(p and p not in result, 'Missing/duplicate pair_id')
        result[p] = r
    return result


def inside(base, relative):
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), 'Unsafe evidence/source path')
    return path


def verify_kit(base, root=None):
    m = json.loads((base / 'manifest.json').read_text())
    require(m['status'] == 'PROVISIONAL_UNVALIDATED' and m['release_blocked'], 'Unexpected kit release status')
    for relative, digest in m['output_sha256'].items():
        require(kit.sha(inside(base, relative)) == digest, f'Kit hash mismatch: {relative}')
    if root is not None:
        for relative, digest in m['input_sha256'].items():
            require(kit.sha(inside(root, relative)) == digest, f'Input hash mismatch: {relative}')
    return m


def roles(raw):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise ValueError('Malformed role_json') from e
    require(isinstance(value, dict) and set(value) == {'roles', 'units'}, 'role_json must contain exactly roles and units')
    for field in ('roles', 'units'):
        require(isinstance(value[field], list) and all(isinstance(x, str) and x.strip() for x in value[field]), 'Malformed role/unit list')
        require(len(value[field]) == len(set(value[field])), 'Duplicate role/unit')
    require(all(r in kit.ROLES for r in value['roles']), 'Unknown evidence role')
    return {k: sorted(v) for k, v in value.items()}


def provenance(row):
    require(row.get('annotator_id', '').strip(), 'Missing annotator provenance')
    require(row.get('provenance_kind') == 'human', 'Only explicitly human provenance is acceptable')
    try:
        t = datetime.fromisoformat(row['annotation_time'].replace('Z', '+00:00'))
        require(t.tzinfo is not None, 'Annotation timestamp needs timezone')
    except (ValueError, KeyError) as e:
        raise ValueError('Invalid annotation timestamp provenance') from e


def validate_rows(rows, expected, base, blank=False):
    require(set(rows) == set(expected), 'Mismatched/missing/extra pair IDs')
    for pair, r in rows.items():
        ref = expected[pair]
        require(set(r) == set(ref), 'Differing sheet columns')
        for key in ref:
            if key not in kit.EDITABLE:
                require(r[key] == ref[key], f'Differing metadata or mismatched query: {pair}: {key}')
        wanted = hashlib.sha256(f'{r["task_id"]}\0{r["doc_id"]}'.encode()).hexdigest()[:24]
        require(pair == wanted, 'Pair ID does not match query/document')
        if blank:
            require(all(r[k] == '' for k in kit.EDITABLE), 'Template contains judgments/provenance')
            continue
        label = r['relevance']
        require(label in VALID, 'Invalid/missing relevance; expected 0/1/2/U')
        provenance(r)
        role = roles(r['role_json'])
        require(r['independent_sufficiency'] in {'YES', 'NO', 'U'}, 'Missing/malformed independent sufficiency')
        require(r['confidence'] in {'LOW', 'MEDIUM', 'HIGH'}, 'Missing/malformed confidence')
        if label == '2':
            require(role['roles'] and role['units'], 'Direct evidence needs role and unit identifiers')
        if r['independent_sufficiency'] == 'YES':
            require(label == '2', 'Sufficiency YES requires direct relevance')
        if label == 'U' or r['confidence'] == 'LOW' or r['independent_sufficiency'] == 'U':
            require(r['notes'].strip(), 'Uncertainty needs explanation')
        span = r['evidence_span']
        require(label != '2' or span.strip(), 'Direct relevance requires evidence span')
        if span:
            ev = json.loads(inside(base, r['evidence_file']).read_text())
            require(ev['doc_id'] == r['doc_id'], 'Evidence document mismatch')
            sections = {s['section_id']: s['text'] for s in ev['sections']}
            require(len(sections) == len(ev['sections']), 'Duplicate evidence section IDs')
            require(r['section_id'] in sections, 'Unknown evidence section')
            try:
                start, end = int(r['start']), int(r['end'])
            except ValueError as e:
                raise ValueError('Malformed evidence offsets') from e
            text = sections[r['section_id']]
            require(0 <= start < end <= len(text) and text[start:end] == span, 'Fabricated/mismatched evidence span or offsets')
        else:
            require(not any(r[k] for k in ('section_id', 'start', 'end')), 'Offsets without evidence')


def cohen_kappa(a, b, binary=False):
    require(len(a) == len(b), 'Unequal label lengths')
    pairs = list(zip(a, b))
    excluded = 0
    if binary:
        excluded = sum('U' in p for p in pairs)
        pairs = [(x == '2', y == '2') for x, y in pairs if 'U' not in (x, y)]
    n = len(pairs)
    if not n:
        return {'value': None, 'n': 0, 'excluded_U_pairs': excluded, 'undefined_reason': 'no eligible pairs'}
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    observed = sum(x == y for x, y in pairs) / n
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return {'value': (observed - expected) / (1 - expected) if expected < 1 else None,
            'n': n, 'excluded_U_pairs': excluded,
            'undefined_reason': 'constant marginal distribution' if expected == 1 else None}


def compare(a, b):
    require(set(a) == set(b) and a, 'Empty or mismatched sheets')
    ids_a = {r['annotator_id'] for r in a.values()}
    ids_b = {r['annotator_id'] for r in b.values()}
    require(len(ids_a) == len(ids_b) == 1 and ids_a.isdisjoint(ids_b), 'Two distinct independent annotator identities required')
    adjudication = []
    for pair in sorted(a):
        x, y = a[pair], b[pair]
        reasons = []
        if any(r['relevance'] == 'U' or r['independent_sufficiency'] == 'U' or r['confidence'] == 'LOW' for r in (x, y)):
            reasons.append('uncertainty_including_U_U')
        for key in ('relevance', 'evidence_span', 'section_id', 'start', 'end', 'independent_sufficiency', 'confidence', 'notes'):
            if x[key] != y[key]:
                reasons.append(key + '_disagreement')
        if roles(x['role_json']) != roles(y['role_json']):
            reasons.append('role_or_unit_disagreement')
        if reasons:
            adjudication.append({'pair_id': pair, 'reasons': reasons,
                                 'adjudicated_relevance': '', 'adjudicator_id': '',
                                 'adjudication_time': '', 'adjudicator_rationale': ''})
    pairs = sorted(a)
    la, lb = [a[p]['relevance'] for p in pairs], [b[p]['relevance'] for p in pairs]
    return {'n_pairs': len(pairs), 'exact_label_agreement': sum(x == y for x, y in zip(la, lb))/len(pairs),
            'four_way_kappa': cohen_kappa(la, lb), 'direct_relevance_kappa': cohen_kappa(la, lb, True),
            'adjudication_required': adjudication, 'final_qrels_exported': False,
            'release_blocked': True, 'note': 'Agreement is not adjudication. No final export is implemented.'}


def validate_structure(items, expected, template=True):
    require(isinstance(items, list) and items, 'Empty structural sheet')
    seen = set()
    for item in items:
        require(set(item) == {'task_id', 'task_family', 'annotation_status', 'required_units', 'evidence_units', 'valid_pairs', 'valid_paths', 'provenance'}, 'Malformed structural fields')
        tid = item['task_id']
        require(tid in expected and tid not in seen, 'Duplicate/unknown structural task')
        seen.add(tid)
        require(item['task_family'] == expected[tid], 'Structural task family mismatch')
        require(item['annotation_status'] in {'unvalidated', 'human_adjudicated', 'expert_adjudicated'}, 'Invalid structural annotation status')
        units = item['required_units']
        require(isinstance(units, list) and all(isinstance(u, str) and u.strip() for u in units) and len(units) == len(set(units)), 'Malformed required_units')
        require(isinstance(item['evidence_units'], dict), 'Malformed evidence_units')
        for doc, values in item['evidence_units'].items():
            require(isinstance(doc, str) and doc and isinstance(values, list) and all(isinstance(v, str) and v in units for v in values), 'Malformed evidence unit assignment')
        for key, minimum in (('valid_pairs', 2), ('valid_paths', 2)):
            require(isinstance(item[key], list), 'Malformed pair/path list')
            for path in item[key]:
                require(isinstance(path, list) and len(path) >= minimum and all(isinstance(d, str) and d in item['evidence_units'] for d in path), 'Malformed pair/path members')
                require(len(set(path)) == len(path) and (key != 'valid_pairs' or len(path) == 2), 'Malformed/duplicate pair/path members')
        require(isinstance(item['provenance'], dict), 'Malformed structural provenance')
        if template:
            require(item['annotation_status'] == 'unvalidated' and not any(item[k] for k in ('required_units', 'evidence_units', 'valid_pairs', 'valid_paths', 'provenance')), 'Structural template must remain unvalidated and empty')
        else:
            raise ValueError('Adjudicated structural import is not enabled: human provenance and task reconciliation required')
    require(seen == set(expected), 'Missing structural tasks')


def verify_task_templates(base, pairs):
    expected = {r['task_id']: {k: r[k] for k in ('task_id', 'task_family', 'question')} for r in pairs.values()}
    for who in ('A', 'B'):
        with (base / f'task_requirements_{who}.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        require(len(rows) == len(expected), 'Empty/incomplete task requirement template')
        seen = set()
        for r in rows:
            tid = r.get('task_id')
            require(tid in expected and tid not in seen, 'Duplicate/mismatched task requirement ID')
            seen.add(tid)
            require(all(r[k] == value for k, value in expected[tid].items()), 'Task requirement metadata mismatch')
            require(all(value == '' for k, value in r.items() if k not in expected[tid]), 'Task template contains judgments')
    return len(expected)


def verify_query_template(root):
    base = root / 'annotations/human_queries_v1'
    m = json.loads((base / 'manifest.json').read_text())
    require(m['status'] == 'blank_template_no_human_data', 'Unexpected human query status')
    require(kit.sha(base / 'human_queries_blank.csv') == m['output_sha256']['human_queries_blank.csv'], 'Query template hash mismatch')
    with (base / 'human_queries_blank.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == m['target_queries'] and 50 <= len(rows) <= 100, 'Invalid blank query quota')
    require(len({r['collection_id'] for r in rows}) == len(rows), 'Duplicate query collection IDs')
    for r in rows:
        require(all(value == '' for k, value in r.items() if k != 'collection_id' and not k.startswith('target_')), 'Query template contains collected data')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, default=kit.ROOT / 'annotations/human_qrel_v2')
    parser.add_argument('--root', type=Path, default=kit.ROOT)
    parser.add_argument('--blank-kit', action='store_true')
    parser.add_argument('--annotator-a', type=Path)
    parser.add_argument('--annotator-b', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    m = verify_kit(args.kit, args.root)
    expected = load(args.kit / 'annotator_A.csv')
    if args.blank_kit:
        require(not args.annotator_a and not args.annotator_b, 'Blank mode cannot accept submitted judgments')
        for who in ('A', 'B'):
            validate_rows(load(args.kit / f'annotator_{who}.csv'), expected, args.kit, blank=True)
        validate_structure(json.loads((args.kit / 'structural_template.json').read_text()), {r['task_id']: r['task_family'] for r in expected.values()})
        task_count = verify_task_templates(args.kit, expected)
        query_count = verify_query_template(args.root)
        report = {'validation': 'blank kit integrity passed', 'pairs': len(expected), 'human_judgments': 0,
                  'task_templates_per_annotator': task_count, 'blank_query_collection_slots': query_count,
                  'release_blocked': True, 'input_sources_hashed': len(m['input_sha256']), 'output_files_hashed': len(m['output_sha256'])}
    else:
        require(args.annotator_a and args.annotator_b, 'Both completed annotator sheets required')
        a, b = load(args.annotator_a), load(args.annotator_b)
        validate_rows(a, expected, args.kit)
        validate_rows(b, expected, args.kit)
        report = compare(a, b)
        report['source_sha256'] = {'A': kit.sha(args.annotator_a), 'B': kit.sha(args.annotator_b), 'kit_manifest': kit.sha(args.kit / 'manifest.json')}
    if args.report:
        require(args.report.resolve().is_relative_to(args.kit.resolve()), 'Report must stay inside v2 kit directory')
        require(not args.report.exists(), 'Refusing to overwrite report')
        kit.dump(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
