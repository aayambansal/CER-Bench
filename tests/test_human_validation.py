"""Synthetic software fixtures only: not human labels or empirical validation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def module(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], SCRIPTS / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b = module('42_prepare_validation_kit.py')
v = module('43_validate_human_labels.py')


def test_identity_aliases_quarantine_every_owner():
    docs = [dict(doc_id='00123', pmid='900', pmcid='123'),
            dict(doc_id='PMC123', pmid='901', pmcid='PMC123'),
            dict(doc_id='902', pmid='902', pmcid='')]
    lookup, bad, conflicts = b.identity_audit(docs)
    assert bad == {0, 1} and 'PMC123' in conflicts
    assert '900' not in lookup and '901' not in lookup
    assert lookup['902'] == 2


def test_numeric_document_id_does_not_guess_namespace():
    assert b.source_urls({'doc_id': '123', 'pmid': '900', 'pmcid': '123'}) == [
        'https://pubmed.ncbi.nlm.nih.gov/900/', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123/']
    assert b.source_urls({'doc_id': '123'}) == []


def test_evidence_untruncated():
    text = 'abc ' * 10000
    assert b.evidence({'abstract': text, 'sections': [{'text': text}]})[0]['text'] == text
    assert b.evidence({'abstract': text, 'sections': [{'text': text}]})[1]['text'] == text


@pytest.fixture
def sample(tmp_path):
    did, tid = '9', 'constraint_1'
    pair = b.hashlib.sha256(f'{tid}\0{did}'.encode()).hexdigest()[:24]
    row = {'pair_id': pair, 'task_id': tid, 'doc_id': did, 'task_family': 'constraint',
           'question': 'What is supported?', 'title': 'Fixture', 'evidence_file': 'evidence.json',
           **{k: '' for k in b.EDITABLE}}
    b.dump(tmp_path / 'evidence.json', {'doc_id': did, 'sections': [{'section_id': 'abstract', 'text': 'Direct support here.'}]})
    complete = dict(row, relevance='2', evidence_span='Direct support', section_id='abstract', start='0', end='14',
                    role_json='{"roles":["constraint"],"units":["C1"]}', independent_sufficiency='YES', confidence='HIGH',
                    annotator_id='synthetic_test_A', annotation_time='2026-09-09T00:00:00+00:00', provenance_kind='human')
    return tmp_path, {pair: row}, {pair: complete}


def test_valid_software_fixture_span(sample):
    base, ref, rows = sample
    v.validate_rows(rows, ref, base)


@pytest.mark.parametrize('key,value', [
    ('question', 'Different query'), ('doc_id', '999'), ('relevance', ''), ('relevance', '3'),
    ('role_json', '[]'), ('role_json', '{"roles":["made_up"],"units":[]}'),
    ('role_json', '{"roles":"constraint","units":["C1"]}'),
    ('evidence_span', 'Fabricated'), ('start', '1'), ('end', '999'), ('section_id', 'missing'),
    ('annotator_id', ''), ('provenance_kind', 'model'), ('annotation_time', 'yesterday'),
    ('independent_sufficiency', ''), ('confidence', '')])
def test_reject_bad_submission(sample, key, value):
    base, ref, rows = sample
    next(iter(rows.values()))[key] = value
    with pytest.raises(ValueError):
        v.validate_rows(rows, ref, base)


def test_empty_and_duplicate_sheets(tmp_path):
    p = tmp_path / 'empty.csv'
    p.write_text('pair_id,relevance\n')
    with pytest.raises(ValueError, match='Empty'):
        v.load(p)
    p.write_text('pair_id,relevance\na,0\na,1\n')
    with pytest.raises(ValueError, match='duplicate'):
        v.load(p)


def test_missing_pair(sample):
    base, ref, _ = sample
    with pytest.raises(ValueError, match='pair IDs'):
        v.validate_rows({}, ref, base)


def test_template_cannot_contain_judgments(sample):
    base, ref, rows = sample
    with pytest.raises(ValueError, match='Template contains'):
        v.validate_rows(rows, ref, base, blank=True)


def test_U_U_requires_adjudication(sample):
    _, _, a = sample
    for r in a.values():
        r.update(relevance='U', independent_sufficiency='U', notes='uncertain')
    c = copy.deepcopy(a)
    next(iter(c.values()))['annotator_id'] = 'synthetic_test_B'
    result = v.compare(a, c)
    assert len(result['adjudication_required']) == 1
    assert result['direct_relevance_kappa']['excluded_U_pairs'] == 1
    assert result['direct_relevance_kappa']['value'] is None
    assert not result['final_qrels_exported']


@pytest.mark.parametrize('key,value', [('role_json', '{"roles":["context"],"units":["C1"]}'),
                                     ('evidence_span', 'Direct'), ('independent_sufficiency', 'NO'),
                                     ('relevance', '1'), ('notes', 'different')])
def test_all_disagreements_adjudicated(sample, key, value):
    _, _, a = sample
    c = copy.deepcopy(a)
    next(iter(c.values())).update(annotator_id='synthetic_test_B', **{key: value})
    assert len(v.compare(a, c)['adjudication_required']) == 1


def test_identical_annotator_rejected(sample):
    _, _, a = sample
    with pytest.raises(ValueError, match='distinct'):
        v.compare(a, a)


def test_kappa_excludes_U_and_degenerate_undefined():
    assert v.cohen_kappa(['2'], ['2'])['value'] is None
    assert v.cohen_kappa([], [])['value'] is None
    k = v.cohen_kappa(['2', '0', 'U', '1'], ['2', '1', '2', 'U'], True)
    assert k['value'] == 1 and k['n'] == 2 and k['excluded_U_pairs'] == 2
    assert v.cohen_kappa(['2', '0'], ['0', '2'], True)['value'] == -1


def test_structural_template_only():
    item = {'task_id': 't', 'task_family': 'multihop', 'annotation_status': 'unvalidated',
            'required_units': [], 'evidence_units': {}, 'valid_pairs': [], 'valid_paths': [], 'provenance': {}}
    v.validate_structure([item], {'t': 'multihop'})
    for key, value in [('required_units', 'bad'), ('evidence_units', []), ('valid_paths', [['missing']]),
                       ('annotation_status', 'human_adjudicated'), ('provenance', {'fake': 'yes'})]:
        bad = dict(item, **{key: value})
        with pytest.raises(ValueError):
            v.validate_structure([bad], {'t': 'multihop'})


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_builder_offline_pool_blinding_hashes_and_quarantine(tmp_path):
    docs = [{'doc_id': str(i), 'pmid': str(i), 'pmcid': '', 'title': f'Fixture {i}', 'abstract': 'Evidence.'} for i in range(1, 12)]
    docs += [dict(docs[0], title='Collision')]
    tasks = [{'task_id': f'{f}_{i}', 'task_family': f, 'question': f'Question {f} {i}', 'supporting_doc_ids': ['2']}
             for f in b.FAMILIES for i in range(14)]
    tasks[0]['supporting_doc_ids'] = ['1']
    tasks += [{'task_id': 'abstention_0', 'task_family': 'abstention', 'question': 'Unanswerable?', 'supporting_doc_ids': []}]
    jsonl(tmp_path / 'data/processed/corpus.jsonl', docs)
    jsonl(tmp_path / 'data/benchmark/test.jsonl', tasks)
    for system in b.SYSTEMS:
        jsonl(tmp_path / f'results/baselines/{system}_test.jsonl', [{'task_id': t['task_id'], 'retrieved_docs': ['3', '1']} for t in tasks])
    m = b.build(tmp_path)
    out = tmp_path / 'annotations/human_qrel_v2'
    assert m['supported_selected'] == 98 and m['excluded_queries'] == 1
    assert m['seed_pairs'] == 97 and m['random_out_of_pool_pairs'] == 98*5
    assert m['quarantined_records'] == 2
    a, c = v.load(out / 'annotator_A.csv'), v.load(out / 'annotator_B.csv')
    assert set(a) == set(c) and list(a) != list(c)
    assert not {'sources', 'rank', 'automatic_label', 'candidate_source'} & set(next(iter(a.values())))
    v.validate_rows(a, a, out, blank=True)
    v.validate_rows(c, a, out, blank=True)
    v.verify_kit(out, tmp_path)
    assert v.verify_task_templates(out, a) == 98
    assert v.verify_query_template(tmp_path) == 80
    prov = json.loads((out / 'coordinator/pool_provenance.json').read_text())
    assert all(p['sources'] == ['uniform_random_out_of_pool'] for p in prov if 'uniform_random_out_of_pool' in p['sources'])
    with pytest.raises(ValueError, match='overwrite'):
        b.build(tmp_path)
    (out / 'annotator_A.csv').write_text('tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        v.verify_kit(out)


def test_legacy_cli_fails_closed():
    old = module('37_merge_human_annotations.py')
    with pytest.raises(SystemExit, match='Legacy merge disabled'):
        old.main()


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        v.inside(tmp_path, '../foreign.json')


def test_doi_alias_quarantine():
    docs = [{'doc_id': '1', 'pmid': '1', 'doi': '10.1234/ABC'},
            {'doc_id': '2', 'pmid': '2', 'doi': 'https://doi.org/10.1234/abc'}]
    lookup, bad, conflicts = b.identity_audit(docs)
    assert not lookup and bad == {0, 1}
    assert 'DOI:10.1234/ABC' in conflicts
