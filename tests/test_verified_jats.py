"""Local source-ownership, rendering, offsets and retrieval-contract tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.corpus.verified_jats import extract_jats, compare_titles, chunk_blocks


def jats(body="<p>Own body.</p>", title="Own study", pmid="1", pmcid="PMC2", abstract="<p>Own abstract.</p>", extra="", ns="", license_url="https://creativecommons.org/licenses/by/4.0/"):
    return f'''<article {ns}><front><article-meta><article-id pub-id-type="pmid">{pmid}</article-id>
    <article-id pub-id-type="pmcid">{pmcid}</article-id><title-group><article-title>{title}</article-title></title-group>
    <abstract>{abstract}</abstract><permissions><copyright-statement>Author copyright</copyright-statement>
    <license xmlns:xlink="http://www.w3.org/1999/xlink" license-type="open-access" xlink:href="{license_url}"><license-p>License conditions apply.</license-p></license></permissions>
    </article-meta></front><body>{body}</body>{extra}</article>'''.encode()


def parse(data, title="Own study"):
    return extract_jats(data, "1", "PMC2", title)


def texts(result):
    return "\n".join(b["text"] for b in result["blocks"])


def test_own_body_only_no_back_or_nested_other_articles():
    r = parse(jats(body='<p>OWN sentence.<xref ref-type="bibr">CITEDMARKER</xref></p><ref-list><ref><p>ILLEGAL BODY REFERENCES</p></ref></ref-list><sub-article><body><p>FOREIGN BODY</p></body></sub-article>',
                   extra='<back><ref-list><ref><mixed-citation>BACK CITATION</mixed-citation></ref></ref-list></back>'))
    assert "OWN sentence." in texts(r)
    assert all(s not in texts(r) for s in ("CITEDMARKER", "ILLEGAL BODY REFERENCES", "FOREIGN BODY", "BACK CITATION"))
    assert not r["coverage"]["unexplained_paragraphs"]
    assert r["coverage"]["explicitly_omitted_paragraphs"] == 2


def test_single_direct_article_wrapper_and_namespaces():
    data = b'<pmc-articleset>' + jats(ns='xmlns="urn:jats"') + b'</pmc-articleset>'
    r = parse(data)
    assert r['selected_root'] == 'article' and r['source_root'] == 'pmc-articleset'
    assert 'Own body.' in texts(r)
    assert r['source_sha256'] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('data', [b'<root><article/></root>', b'<pmc-articleset><article/><article/></pmc-articleset>', b'<pmc-articleset><note/><article/></pmc-articleset>', b'<article><front/><front/></article>'])
def test_reject_nonarticle_or_ambiguous_roots(data):
    with pytest.raises(ValueError):
        parse(data)


@pytest.mark.parametrize('data', [jats(pmid='9'), jats(pmcid='PMC9'), jats(pmid=''), jats().replace(b'<article-id pub-id-type="pmid">1</article-id>', b'<article-id pub-id-type="pmid">1</article-id><article-id pub-id-type="pmid">9</article-id>')])
def test_exact_own_id_gate(data):
    with pytest.raises(ValueError, match='PMID/PMCID'):
        parse(data)


def test_title_policy_punctuation_subtitle_and_major_mismatch():
    assert compare_titles('Own study: Some findings.', 'OWN STUDY — Some findings')['accepted']
    data = jats().replace(b'</article-title>', b'</article-title><subtitle>Some findings</subtitle>')
    assert parse(data, 'Own study: Some findings.')['title_comparison']['normalized_equal']
    bad = parse(jats(title='An unrelated paper on other subjects'))
    assert bad['status'] == 'major_title_mismatch_quarantined' and bad['blocks'] == []


def test_nested_sections_lists_and_statement_not_duplicated():
    data = jats(body='<sec sec-type="methods"><title>Methods</title><p>PARA_ONE <italic>mixed</italic> tail.</p><list><list-item><p>LIST_ONE</p><list><list-item><p>NESTED_ONE</p></list-item></list></list-item></list><statement><p>STATEMENT_ONE</p></statement><sec><title>Nested</title><p>PARA_TWO</p></sec></sec>')
    r = parse(data)
    for marker in ['PARA_ONE', 'LIST_ONE', 'NESTED_ONE', 'STATEMENT_ONE', 'PARA_TWO']:
        assert texts(r).count(marker) == 1
    assert 'PARA_ONE mixed tail.' in texts(r)
    assert r['coverage']['source_paragraphs'] == r['coverage']['handled_paragraphs'] == 6
    assert not r['coverage']['duplicate_paragraph_visits']
    assert any(b['kind'] == 'statement_paragraph' for b in r['blocks'])
    assert any(s['parent_section_id'] is not None for s in r['sections'])


def test_paragraph_nested_block_order_and_no_duplicate_caption():
    r = parse(jats(body='<p>Before <fig><caption><p>CAPTION_ONCE</p></caption><graphic href="image"/></fig> After.</p>'))
    rendered = texts(r)
    assert rendered.index('Before') < rendered.index('CAPTION_ONCE') < rendered.index('After')
    assert rendered.count('CAPTION_ONCE') == 1
    assert any(o['reason'] == 'image_not_extracted' for o in r['omissions'])


def test_table_row_column_boundaries_and_spans():
    r = parse(jats(body='<table-wrap><caption><p>TABLE_CAPTION</p></caption><table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td rowspan="2">X</td><td><p>Y</p><p>Z</p></td></tr><tr><td colspan="2">W</td></tr></tbody></table><table-wrap-foot><fn><p>TABLE_NOTE</p></fn></table-wrap-foot></table-wrap>'))
    rows = [b for b in r['blocks'] if b['kind'] == 'table_row']
    assert len(rows) == 3
    assert rows[0]['text'] == 'row 1\tcol 1=A\tcol 2=B'
    assert rows[1]['table_cells'][0]['rowspan'] == 2
    assert rows[2]['table_cells'][0]['column'] == 1
    assert rows[2]['table_cells'][0]['colspan'] == 2
    assert texts(r).count('TABLE_CAPTION') == texts(r).count('TABLE_NOTE') == 1
    assert not r['coverage']['unexplained_paragraphs']


def test_formula_alternatives_once_images_supplements_marked():
    r = parse(jats(body='<p>Equation <inline-formula><alternatives><tex-math>x+y</tex-math><m:math xmlns:m="http://www.w3.org/1998/Math/MathML"><m:mi>x</m:mi><m:mo>+</m:mo><m:mi>y</m:mi></m:math><graphic href="formula.png"/></alternatives></inline-formula>.</p><disp-formula><graphic href="only_image.png"/></disp-formula><supplementary-material><p>SUPPLEMENT_TEXT</p></supplementary-material>'))
    assert texts(r).count('x+y') == 1
    assert 'SUPPLEMENT_TEXT' not in texts(r)
    reasons = {o['reason'] for o in r['omissions']}
    assert {'formula_alternative_not_duplicated', 'formula_without_extractable_text', 'supplement_not_extracted'} <= reasons


def test_mathml_linearization_explicit_limit():
    r = parse(jats(body='<disp-formula><m:math xmlns:m="http://www.w3.org/1998/Math/MathML"><m:msub><m:mi>x</m:mi><m:mn>2</m:mn></m:msub></m:math></disp-formula>'))
    assert 'x 2' in texts(r)
    assert any(o['reason'] == 'mathml_linearized_structure_loss' for o in r['omissions'])


def test_mathml_annotation_not_duplicated_or_flattened():
    r = parse(jats(body='<disp-formula><m:math xmlns:m="http://www.w3.org/1998/Math/MathML"><m:semantics><m:mi>x</m:mi><m:annotation>NOT_INDEXED_ALTERNATE</m:annotation></m:semantics></m:math></disp-formula>'))
    assert 'NOT_INDEXED_ALTERNATE' not in texts(r)
    assert any(o['reason'] == 'mathml_annotation_or_unsupported_structure_omitted' for o in r['omissions'])


def test_nested_table_lists_and_bare_math_are_rendered_once():
    r = parse(jats(body='<table-wrap><table><tr><td><list><list-item><p>CELL_ONE</p><list><list-item><p>CELL_TWO</p></list-item></list></list-item></list><m:math xmlns:m="http://www.w3.org/1998/Math/MathML"><m:mi>x</m:mi></m:math></td></tr></table></table-wrap>'))
    assert texts(r).count('CELL_ONE') == texts(r).count('CELL_TWO') == 1
    assert not r['coverage']['unexplained_paragraphs']
    assert not r['coverage']['duplicate_paragraph_visits']


def test_unknown_structures_are_omitted_and_paragraph_accounted():
    r = parse(jats(body='<unknown-container><p>NOT_ALLOWED</p></unknown-container><p>ALLOWED</p>'))
    assert 'NOT_ALLOWED' not in texts(r)
    assert r['coverage']['explicitly_omitted_paragraphs'] == 1
    assert not r['coverage']['unexplained_paragraphs']


def test_license_scope_strings_links_and_no_blanket_rights():
    lic = parse(jats())['license']
    assert lic['category'] == 'machine_readable_cc_license'
    assert lic['machine_readable_licenses'] == ['CC-BY-4.0']
    assert lic['permissions'][0]['scope'] == 'own_article/front/article-meta/permissions'
    assert 'Author copyright' in lic['permissions'][0]['text']
    assert 'http://www.w3.org/1999/xlink' in lic['permissions'][0]['xml']
    assert lic['public_redistribution_authorized'] is False
    unknown = parse(jats(license_url='https://example.org/unknown-license'))['license']
    assert unknown['category'] == 'license_unreviewed'


@pytest.mark.parametrize('lengths', [[5], [5000], [10000, 30], [2000, 2000, 1000], [100, 100, 15000]])
def test_chunk_codepoint_bounds_overlap_full_coverage_and_determinism(lengths):
    blocks = [dict(block_id=f'b{i}', text=('β😀 word ' * (n//8+1))[:n], xml_path=f'/article/body/p[{i+1}]',
                   source_sha256='a'*64, section_id='sec') for i,n in enumerate(lengths)]
    doc, chunks = chunk_blocks('PMID:1', blocks)
    assert (doc, chunks) == chunk_blocks('PMID:1', [dict(b) for b in blocks])
    end = 0
    for c in chunks:
        assert 0 < len(c['text']) <= 4096
        assert c['text'] == doc[c['document_start']:c['document_end']]
        assert c['doc_id'] == c['article_id'] == 'PMID:1'
        assert c['document_start'] <= end
        assert end-c['document_start'] <= 256
        end = c['document_end']
        for span in c['source_spans']:
            b = next(b for b in blocks if b['block_id'] == span['block_id'])
            assert c['text'][span['chunk_start']:span['chunk_end']] == b['text'][span['block_start']:span['block_end']]
    assert end == len(doc)


def builder():
    spec = importlib.util.spec_from_file_location('restore50', BASE / 'scripts/50_restore_verified_fulltext.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_inputs(base):
    from src.corpus.pubmed_xml import parse_records, inspect_fulltext
    b = builder()
    raw = base / 'data/raw/pubmed_verified_v1'
    raw.mkdir(parents=True)
    source = raw / 'batch.xml'
    source.write_text('<PubmedArticleSet>' + ''.join(f'<PubmedArticle><MedlineCitation><PMID>{i}</PMID><Article><ArticleTitle>Own study</ArticleTitle>' +
                 ('<Abstract><AbstractText>Public abstract.</AbstractText></Abstract>' if i==1 else '') +
                 '</Article></MedlineCitation><PubmedData><ArticleIdList>' + (f'<ArticleId IdType="pmc">PMC{i+1}</ArticleId>' if i < 3 else '') +
                 '</ArticleIdList></PubmedData></PubmedArticle>' for i in (1,2,3)) + '</PubmedArticleSet>')
    rows = parse_records(source.read_bytes())
    ft = base / 'data/raw/fulltext'
    ft.mkdir(parents=True)
    for d in rows:
        d.update(doc_id=d['pmid'], provenance=dict(raw_path=str(source.relative_to(base)), raw_sha256=b.sha(source)),
                 fulltext_status='no_matching_local_front', fulltext_candidates=[])
        if d['pmid'] in {'1','2'}:
            p = ft / f'{d["pmid"]}.xml'
            p.write_bytes(jats(pmid=d['pmid'], pmcid=d['pmcid']))
            info = inspect_fulltext(p.read_bytes())
            info.update(path=str(p.relative_to(base)), sha256=b.sha(p))
            d.update(fulltext_status='identity_matched_raw_candidate_text_unvalidated', fulltext_candidates=[info])
    corpus = base / 'data/processed/authoritative_v1/corpus.jsonl'
    corpus.parent.mkdir(parents=True)
    b.write_rows(corpus, rows)
    m = base / 'results/readiness/authoritative_corpus/output_manifest.json'
    m.parent.mkdir(parents=True)
    b.write_json(m, [dict(path=str(corpus.relative_to(base)), sha256=b.sha(corpus))])
    for rel in ['src/corpus/verified_jats.py', 'src/corpus/pubmed_xml.py', 'scripts/50_restore_verified_fulltext.py']:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((BASE / rel).read_bytes())
    return b


def test_two_independent_builds_and_worker49_interface(tmp_path):
    b = fixture_inputs(tmp_path)
    out1, out2 = tmp_path/'one', tmp_path/'two'
    b.build(tmp_path, out1)
    b.build(tmp_path, out2)
    assert {p.name:b.sha(p) for p in out1.iterdir()} == {p.name:b.sha(p) for p in out2.iterdir()}
    from src.retrieval.verified_bm25 import validate_inputs
    docs, chunks, manifest, _ = validate_inputs(out1/'corpus.jsonl', out1/'chunks.jsonl', out1/'dataset_manifest.json')
    assert len(docs) == 3
    assert all(d['doc_id'] == d['article_id'] == 'PMID:'+d['pmid'] for d in docs)
    assert docs[2]['title_only'] and not docs[2]['has_fulltext']
    assert docs[1]['has_fulltext'] and not docs[1]['abstract']
    assert manifest['human_validation'] is False
    with pytest.raises(FileExistsError):
        b.build(tmp_path, out1)


def test_changed_raw_identity_is_quarantined_not_restored(tmp_path):
    b = fixture_inputs(tmp_path)
    (tmp_path/'data/raw/fulltext/1.xml').write_bytes(jats(pmid='999'))
    out = tmp_path/'out'
    b.build(tmp_path, out)
    docs = b.read_rows(out/'corpus.jsonl')
    assert not docs[0]['has_fulltext']
    assert docs[0]['fulltext_status'] == 'candidate_quarantined_error'
    assert docs[0]['text'] == 'Own study\n\nPublic abstract.'


def test_modified_authoritative_corpus_fails_without_output(tmp_path):
    b = fixture_inputs(tmp_path)
    p = tmp_path/'data/processed/authoritative_v1/corpus.jsonl'
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match='checksum'):
        b.build(tmp_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()
