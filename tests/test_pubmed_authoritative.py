"""Offline regression tests: no public requests or optional dependencies."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.corpus.pubmed_xml import parse_pubmed_xml, parse_records, inspect_fulltext


def article(pmid="1", own="", references="", title="A <i>mixed</i> title", extra=""):
    return f'''<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>
    <ArticleTitle>{title}</ArticleTitle><Abstract><AbstractText Label="AIM">A <b>bold</b> result.</AbstractText></Abstract>
    <Journal><JournalIssue><PubDate><MedlineDate>2020 Dec-2021 Jan</MedlineDate></PubDate></JournalIssue></Journal>
    {extra}</Article></MedlineCitation><PubmedData><ArticleIdList>
    <ArticleId IdType="pubmed">{pmid}</ArticleId>{own}</ArticleIdList>
    <ReferenceList><Reference><ArticleIdList>{references}</ArticleIdList></Reference></ReferenceList>
    </PubmedData></PubmedArticle>'''


def xml(*articles):
    return "<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"


def test_cited_ids_never_supply_own_pmc_or_doi():
    r = parse_records(xml(article(references='<ArticleId IdType="pmc">PMC6286148</ArticleId><ArticleId IdType="doi">10.cited</ArticleId>')))[0]
    assert r["pmcid"] == r["doi"] == ""
    assert r["pmid"] == "1"
    assert r["title"] == "A mixed title"
    assert r["abstract"] == "AIM: A bold result."
    assert r["year"] == 2020


def test_own_id_priority_and_elocation_fallback():
    own = '<ArticleId IdType="pmc">PMC12</ArticleId><ArticleId IdType="doi">10.own</ArticleId>'
    ref = '<ArticleId IdType="pmc">PMC99</ArticleId><ArticleId IdType="doi">10.ref</ArticleId>'
    e = '<ELocationID EIdType="doi">10.elocation</ELocationID>'
    r = parse_records(xml(article(own=own, references=ref, extra=e)))[0]
    assert (r["pmcid"], r["doi"]) == ("PMC12", "10.own")
    assert parse_records(xml(article(references=ref, extra=e)))[0]["doi"] == "10.elocation"
    assert parse_records(xml(article(extra='<ELocationID EIdType="doi" ValidYN="N">bad</ELocationID>')))[0]["doi"] == ""


def test_namespace_and_nested_fake_article():
    doc = xml(article(references='<PubmedArticle><MedlineCitation><PMID>999</PMID></MedlineCitation></PubmedArticle>'))
    doc = doc.replace('<PubmedArticleSet>', '<PubmedArticleSet xmlns="urn:ncbi:test">')
    result = parse_pubmed_xml(doc, ["1", "2"])
    assert [r["pmid"] for r in result["records"]] == ["1"]
    assert result["missing_pmids"] == ["2"]


def test_books_missing_article_and_not_returned_explicit():
    book = '<PubmedBookArticle><BookDocument><PMID>2</PMID><ArticleTitle>Book chapter</ArticleTitle></BookDocument></PubmedBookArticle>'
    broken = '<PubmedArticle><MedlineCitation><PMID>3</PMID></MedlineCitation></PubmedArticle>'
    noid = '<PubmedArticle><MedlineCitation><Article/></MedlineCitation></PubmedArticle>'
    result = parse_pubmed_xml(xml(article(), book, broken, noid), ["1", "2", "3", "4"])
    assert result["returned_pmids"] == ["1", "2", "3"]
    assert result["missing_pmids"] == ["4"]
    assert len(result["problems"]) == 3
    assert result["problems"][0]["reason"] == "unsupported_record_type"
    assert [r["pmid"] for r in result["records"]] == ["1"]
    with pytest.raises(ValueError):
        parse_records(xml(book))


def test_duplicate_records_requests_and_unexpected_ids():
    result = parse_pubmed_xml(xml(article(), article(), article("3")), ["1", "1", "2"])
    assert result["duplicate_pmids"] == ["1"]
    assert result["duplicate_requested_pmids"] == ["1"]
    assert result["unexpected_pmids"] == ["3"]
    assert result["missing_pmids"] == ["2"]
    assert [r["pmid"] for r in result["records"]] == ["3"]


def test_conflicting_own_ids_fail_closed_but_repeated_identical_ids_ok():
    own = '<ArticleId IdType="pmc">PMC12</ArticleId>'
    assert parse_records(xml(article(own=own + own)))[0]["pmcid"] == "PMC12"
    result = parse_pubmed_xml(xml(article(own=own + '<ArticleId IdType="pmc">PMC13</ArticleId>')))
    assert not result["records"]
    assert 'conflicting_pmc' in result["problems"][0]["issues"]
    result = parse_pubmed_xml(xml(article(own='<ArticleId IdType="pubmed">2</ArticleId>')))
    assert not result["records"]


@pytest.mark.parametrize("doc", ["<html/>", "<ERROR>bad request</ERROR>", "<PubmedArticleSet>"])
def test_invalid_response_not_a_success(doc):
    with pytest.raises(Exception):
        parse_pubmed_xml(doc)


def jats(pmid="1", pmc="PMC12", extra="", ns=""):
    return f'''<article {ns}><front><article-meta><article-id pub-id-type="pmid">{pmid}</article-id>
    <article-id pub-id-type="pmcid">{pmc}</article-id>{extra}<permissions>
    <license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://example.org/license">Restricted license</license>
    </permissions></article-meta></front><body><p>Body should not be extracted.</p></body>
    <back><ref-list><ref><article-id pub-id-type="pmid">99</article-id></ref></ref-list></back></article>'''


def test_jats_direct_front_namespace_and_license():
    result = inspect_fulltext(jats(ns='xmlns="urn:jats"'))
    assert result["pmids"] == ["1"] and result["pmcids"] == ["PMC12"]
    assert result["status"] == "front_identity_candidate"
    assert result["namespace"] == "urn:jats"
    assert 'http://www.w3.org/1999/xlink' in result["license_xml"][0]
    assert 'Restricted license' in result["license_xml"][0]
    assert result["redistribution_status"] == "not_assessed_no_rights_granted"
    assert "body" not in result


def test_jats_missing_conflicting_ids_multiple_articles():
    assert inspect_fulltext(jats(pmid=""))["status"] == "missing_or_conflicting_front_ids"
    assert inspect_fulltext(jats(extra='<article-id pub-id-type="pmid">2</article-id>'))["status"] == "missing_or_conflicting_front_ids"
    assert inspect_fulltext('<pmc-articleset>' + jats() + jats() + '</pmc-articleset>')["status"] == "unsupported_or_multiple_articles"
    assert inspect_fulltext('<article><back><article-id pub-id-type="pmid">1</article-id></back></article>')["status"] == "missing_or_conflicting_front_ids"


def builder():
    spec = importlib.util.spec_from_file_location("build47", BASE / "scripts/47_build_authoritative_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_base(tmp_path, duplicate_local=False):
    b = builder()
    b.write_rows(tmp_path / 'data/processed/corpus.jsonl', [dict(pmid="1", doc_id="PMC99", pmcid="PMC99")])
    p = tmp_path / b.RAW / 'pubmed_batch_001.xml'
    p.parent.mkdir(parents=True)
    p.write_text(xml(article(own='<ArticleId IdType="pmc">PMC12</ArticleId>')))
    b.write_json(Path(str(p) + '.manifest.json'), dict(filename=p.name, sha256=b.sha(p), bytes=p.stat().st_size,
                 requested_pmids=["1"], returned_pmids=["1"], missing_pmids=[],
                 url=b.URL + '1&retmode=xml', retrieved_at_utc="2026-09-09T00:00:00+00:00"))
    ft = tmp_path / 'data/raw/fulltext'
    ft.mkdir(parents=True)
    (ft / 'misleading_name.xml').write_text(jats())
    if duplicate_local:
        (ft / 'another_name.xml').write_text(jats())
    return b


def test_build_canonical_identity_no_fulltext_or_gold_restoration(tmp_path):
    b = fixture_base(tmp_path)
    assert b.build(tmp_path) == 0
    doc = b.rows(tmp_path / b.OUT / 'corpus.jsonl')[0]
    assert doc['doc_id'] == '1' and doc['pmcid'] == 'PMC12'
    assert doc['fulltext_status'] == 'identity_matched_raw_candidate_text_unvalidated'
    assert doc['has_fulltext'] is False and doc['sections'] == []
    assert b.rows(tmp_path / b.OUT / 'document_task_eligibility.jsonl')[0]['eligible_as_gold'] is False


def test_local_duplicate_files_not_unique_match(tmp_path):
    b = fixture_base(tmp_path, duplicate_local=True)
    assert b.build(tmp_path) == 0
    assert b.rows(tmp_path / b.OUT / 'corpus.jsonl')[0]['fulltext_status'] == 'ambiguous_local_or_authoritative_identity'


def test_incomplete_build_fails_release_and_lists_missing(tmp_path):
    b = builder()
    b.write_rows(tmp_path / 'data/processed/corpus.jsonl', [dict(pmid='1', doc_id='1')])
    assert b.build(tmp_path) == 2
    summary = json.loads((tmp_path / b.REPORT / 'summary.json').read_text())
    assert summary['metadata_release_pass'] is False
    assert summary['missing_or_unsupported_pmids'] == ['1']


def test_checksum_tampering_fails_release(tmp_path):
    b = fixture_base(tmp_path)
    (tmp_path / b.RAW / 'pubmed_batch_001.xml').write_text(xml(article('2')))
    assert b.build(tmp_path) == 2


def test_import_parser_and_client_without_optional_packages_or_secrets():
    code = '''import sys, os
sys.path.insert(0, sys.argv[1])
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('Bio', 'yaml'):
            raise AssertionError('optional dependency imported: ' + fullname)
sys.meta_path.insert(0, Block())
class NoSecrets(dict):
    def get(self, *args): raise AssertionError('environment read')
    def __getitem__(self, key): raise AssertionError('environment read')
os.environ = NoSecrets()
from src.corpus.pubmed_client import _parse_pubmed_xml
assert _parse_pubmed_xml('<PubmedArticleSet/>') == []
'''
    result = subprocess.run([sys.executable, '-S', '-c', code, str(BASE)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_pilot_regression_if_available():
    path = BASE / 'data/raw/pubmed_verified_v1/identity_pilot.xml'
    if not path.exists():
        pytest.skip('public pilot not installed')
    records = parse_records(path.read_bytes())
    assert {r['pmid'] for r in records} == {'38308006', '30610625', '40828286'}
    assert all(not r['pmcid'] for r in records)


def test_wrong_local_front_not_restored_even_if_filename_matches(tmp_path):
    b = fixture_base(tmp_path)
    ft = tmp_path / 'data/raw/fulltext'
    (ft / 'misleading_name.xml').unlink()
    (ft / 'PMC12.xml').write_text(jats(pmid='999', pmc='PMC12'))
    assert b.build(tmp_path) == 0
    doc = b.rows(tmp_path / b.OUT / 'corpus.jsonl')[0]
    assert doc['fulltext_status'] == 'no_matching_local_front'
    assert not doc['has_fulltext'] and not doc['fulltext_candidates']


def test_cross_batch_duplicate_fails_release(tmp_path):
    b = fixture_base(tmp_path)
    first = tmp_path / b.RAW / 'pubmed_batch_001.xml'
    second = first.with_name('pubmed_batch_002.xml')
    second.write_bytes(first.read_bytes())
    m = json.loads(Path(str(first) + '.manifest.json').read_text())
    m['filename'] = second.name
    b.write_json(Path(str(second) + '.manifest.json'), m)
    assert b.build(tmp_path) == 2


def test_complete_public_snapshot_independently_matches_direct_ids():
    import hashlib
    import xml.etree.ElementTree as ET
    from src.corpus.pubmed_xml import normalize_pmc, text
    corpus = BASE / 'data/processed/authoritative_v1/corpus.jsonl'
    if not corpus.exists():
        pytest.skip('public snapshot not installed')
    docs = {r['pmid']: r for r in builder().rows(corpus)}
    observed = []
    for path in sorted((BASE / 'data/raw/pubmed_verified_v1').glob('pubmed_batch_*.xml.manifest.json')):
        m = json.loads(path.read_text())
        data = path.with_name(m['filename']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == m['sha256']
        root = ET.fromstring(data)
        assert not root.findall('PubmedBookArticle')
        for node in root.findall('PubmedArticle'):
            pmid = node.findtext('MedlineCitation/PMID')
            observed.append(pmid)
            own_pmc = {normalize_pmc(e.text) for e in node.findall('PubmedData/ArticleIdList/ArticleId') if e.get('IdType') == 'pmc'}
            assert own_pmc == ({docs[pmid]['pmcid']} if docs[pmid]['pmcid'] else set())
            assert docs[pmid]['title'] == text(node.find('MedlineCitation/Article/ArticleTitle'))
            assert docs[pmid]['doc_id'] == pmid
            assert docs[pmid]['sections'] == [] and docs[pmid]['has_fulltext'] is False
    assert len(observed) == len(set(observed)) == len(docs) == 4936
    assert sum(bool(r['pmcid']) for r in docs.values()) == 2641
