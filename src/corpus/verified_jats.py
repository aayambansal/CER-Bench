"""Conservative local JATS extraction. No network or optional dependencies.

Offsets are Unicode codepoints in rendered source blocks/document text, NOT XML
byte offsets. XPath-like source locators and raw XML hashes bind the rendering.
Only a single direct article (optionally in a single-article pmc-articleset) is
accepted. Unknown structures are explicitly omitted, not flattened wholesale.
"""
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET

CORPUS_ID = "cerbench-authoritative-fulltext-v1"
INLINE = {"italic", "bold", "sup", "sub", "underline", "monospace", "sc",
          "styled-content", "named-content", "ext-link", "uri", "email", "label",
          "strike", "overline", "abbrev", "ruby", "rt", "xref"}
CONTAINERS = {"list", "list-item", "statement", "disp-quote", "boxed-text",
              "def-list", "def-item", "def", "table-wrap-foot", "fn"}
EXCLUDE = {"back", "ref-list", "ref", "element-citation", "mixed-citation",
           "sub-article", "response", "article", "related-article"}
MEDIA = {"graphic", "inline-graphic", "media", "supplementary-material",
         "inline-supplementary-material"}
INLINE_CONTAINERS = {"p", "list", "list-item", "statement", "def-list", "def-item", "term", "def", "disp-quote"}
MATHML = {"math", "mrow", "mi", "mn", "mo", "mtext", "msub", "msup", "msubsup", "mfrac", "msqrt", "mroot",
          "mfenced", "mover", "munder", "munderover", "mstyle", "mtable", "mtr", "mtd", "mspace", "maligngroup", "semantics"}


def name(e):
    return e.tag.rsplit("}", 1)[-1]


def children(e, tag):
    return [c for c in e if name(c) == tag]


def child(e, tag):
    return next(iter(children(e, tag)), None)


def clean(value):
    return " ".join(value.split())


def own_text(e):
    return clean("".join(e.itertext())) if e is not None else ""


def title_normalize(value):
    # Ignore punctuation, whitespace and case, but keep letters/numbers/order.
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def compare_titles(authoritative, local):
    a, b = title_normalize(authoritative), title_normalize(local)
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 0
    ok = bool(a and b) and (a == b or (ratio >= .90 and jaccard >= .75))
    return dict(authoritative_title=authoritative, local_title=local,
                normalized_equal=bool(a) and a == b, sequence_ratio=ratio,
                token_jaccard=jaccard, accepted=ok,
                policy="NFKC_casefold_punctuation_space; exact_or_sequence>=0.90_and_token_jaccard>=0.75")


def article_root(data):
    root = ET.fromstring(data)
    wrapper = name(root)
    if wrapper == "pmc-articleset":
        if len(root) != 1 or name(root[0]) != "article":
            raise ValueError("pmc-articleset must contain exactly one direct article and no other elements")
        article = root[0]
    elif wrapper == "article":
        article = root
    else:
        raise ValueError("root must be article or a single-direct-article pmc-articleset")
    if name(article) != "article":
        raise ValueError("selected root is not exactly article")
    if len(children(article, "front")) != 1:
        raise ValueError("article must have exactly one front")
    front = child(article, "front")
    if len(children(front, "article-meta")) != 1 or len(children(article, "body")) > 1:
        raise ValueError("ambiguous article-meta or body")
    return root, article, child(front, "article-meta"), wrapper


def permissions(meta):
    """Classify explicit CC URLs, not prose guesses; no general rights grant."""
    entries, allowed = [], set()
    for scope in children(meta, "permissions"):
        links = sorted({v for e in scope.iter() for k, v in e.attrib.items()
                        if k.rsplit("}", 1)[-1] == "href"})
        licenses = []
        for lic in children(scope, "license"):
            license_links = sorted({v for e in lic.iter() for k, v in e.attrib.items()
                                    if k.rsplit("}", 1)[-1] == "href"})
            for url in license_links:
                m = re.fullmatch(r"https?://creativecommons\.org/(licenses/(by(?:-nc)?(?:-sa|-nd)?)/[1-4]\.0|publicdomain/zero/1\.0)/?(?:legalcode)?", url)
                if m:
                    allowed.add("CC0-1.0" if "publicdomain" in url else "CC-" + m.group(2).upper() + "-" + url.split(m.group(2) + "/", 1)[1].split("/")[0])
            licenses.append(dict(text=own_text(lic), attributes=dict(lic.attrib), links=license_links,
                                 xml=ET.tostring(lic, encoding="unicode")))
        entries.append(dict(scope="own_article/front/article-meta/permissions", text=own_text(scope), links=links,
                            licenses=licenses, xml=ET.tostring(scope, encoding="unicode")))
    category = "machine_readable_cc_license" if len(allowed) == 1 else "conflicting_machine_readable_licenses" if len(allowed) > 1 else "license_unreviewed"
    return dict(category=category, machine_readable_licenses=sorted(allowed), permissions=entries,
                use_scope="local_research_only", public_redistribution_authorized=False,
                limits="CC URL recognition only; conditions, third-party assets and other scopes not adjudicated")


class Extractor:
    def __init__(self, root, article, source_hash):
        self.article = article
        self.hash = source_hash
        self.paths = {}
        self.blocks, self.omissions, self.sections = [], [], []
        self.handled_paragraphs = set()
        self.paragraph_visits = Counter()
        self.paragraphs_with_text = set()
        def locate(e, path):
            self.paths[id(e)] = path
            counts = Counter()
            for c in e:
                counts[name(c)] += 1
                locate(c, path + f"/{name(c)}[{counts[name(c)]}]")
        locate(root, f"/{name(root)}[1]")

    def omit(self, e, reason):
        self.omissions.append(dict(xml_path=self.paths[id(e)], element=name(e), reason=reason))

    def inline(self, e):
        tag = name(e)
        if tag in EXCLUDE:
            self.omit(e, "references_or_other_article_excluded")
            return ""
        if tag in MEDIA:
            self.omit(e, "supplement_not_extracted" if "supplement" in tag or tag == "media" else "image_not_extracted")
            return ""
        if tag == "xref" and e.get("ref-type") == "bibr":
            self.omit(e, "bibliographic_citation_marker_excluded")
            return ""
        if tag in {"inline-formula", "disp-formula"}:
            return self.formula(e)
        if tag == "math":
            self.omit(e, "mathml_linearized_structure_loss")
            return self.math_text(e)
        if tag in {"break", "hr"}:
            return "\n"
        if tag == "p":
            self.handled_paragraphs.add(self.paths[id(e)])
            self.paragraph_visits[self.paths[id(e)]] += 1
        output = ["\n- " if tag == "list-item" else "", e.text or ""]
        for c in e:
            ct = name(c)
            if ct in INLINE | MEDIA | EXCLUDE | INLINE_CONTAINERS | {"inline-formula", "disp-formula", "math", "break", "hr"}:
                output.append(self.inline(c))
                if ct in INLINE_CONTAINERS:
                    output.append("\n")
            else:
                self.omit(c, "unsupported_inline_element")
            output.append(c.tail or "")
        result = "".join(output)
        if tag == "p" and clean(result):
            self.paragraphs_with_text.add(self.paths[id(e)])
        return result

    def math_text(self, e):
        if name(e) not in MATHML:
            self.omit(e, "mathml_annotation_or_unsupported_structure_omitted")
            return ""
        values = [e.text or ""]
        for c in e:
            values.append(self.math_text(c))
            values.append(c.tail or "")
        return clean(" ".join(values))

    def formula(self, e):
        self.omit(e, "formula_rendering_not_semantically_validated")
        options = list(e)
        for alt in children(e, "alternatives"):
            options.extend(alt)
        tex = next((c for c in options if name(c) == "tex-math"), None)
        math = next((c for c in options if name(c) == "math"), None)
        if tex is not None or math is not None:
            for c in options:
                if name(c) in MEDIA:
                    self.omit(c, "formula_image_alternative_not_extracted")
        if tex is not None:
            if math is not None:
                self.omit(math, "formula_alternative_not_duplicated")
            for c in tex:
                self.omit(c, "tex_math_nested_markup_omitted")
            return clean(tex.text or "")
        if math is not None:
            self.omit(math, "mathml_linearized_structure_loss")
            # MathML leaf content only, separated to avoid merging variables.
            return self.math_text(math)
        fallback = clean(e.text or "")
        for c in e:
            if name(c) in INLINE:
                fallback += " " + clean(self.inline(c))
            else:
                self.omit(c, "formula_representation_omitted")
        if not fallback:
            self.omit(e, "formula_without_extractable_text")
        return fallback.strip()

    def section(self, e, parent, kind, heading=""):
        s = dict(section_id=f"section:{len(self.sections):05d}", parent_section_id=parent,
                 section_type=kind, heading=heading, xml_path=self.paths[id(e)])
        self.sections.append(s)
        return s

    def emit(self, value, e, kind, section, **extra):
        value = value.strip()
        if not value:
            return
        self.blocks.append(dict(block_id=f"jats:{len(self.blocks):06d}", text=value, kind=kind,
                                section_id=section["section_id"], section_type=section["section_type"],
                                section_heading=section["heading"], xml_path=self.paths[id(e)],
                                source_sha256=self.hash, source_type="local_jats", **extra))

    def paragraph(self, e, section, kind="paragraph"):
        path = self.paths[id(e)]
        self.handled_paragraphs.add(path)
        self.paragraph_visits[path] += 1
        run = [e.text or ""]
        emitted_before = len(self.blocks)
        for c in e:
            if name(c) in INLINE | MEDIA | EXCLUDE | {"inline-formula", "break", "hr"}:
                run.append(self.inline(c))
            else:
                self.emit(clean("".join(run)), e, kind, section)
                run = []
                self.walk(c, section)
            run.append(c.tail or "")
        self.emit(clean("".join(run)), e, kind, section)
        if len(self.blocks) > emitted_before:
            self.paragraphs_with_text.add(path)

    def table(self, e, section):
        groups = []
        for c in e:
            if name(c) == "tr":
                groups.append(("table", c))
            elif name(c) in {"thead", "tbody", "tfoot"}:
                groups.extend((name(c), r) for r in children(c, "tr"))
            elif name(c) not in {"col", "colgroup"}:
                self.omit(c, "unsupported_table_structure")
        occupied = {}
        for ri, (group, row) in enumerate(groups):
            cells, col = [], 0
            for c in row:
                if name(c) not in {"td", "th"}:
                    self.omit(c, "unsupported_table_row_child")
                    continue
                while occupied.get(col, -1) >= ri:
                    col += 1
                try:
                    rowspan = int(c.get("rowspan", "1")); colspan = int(c.get("colspan", "1"))
                    if not 1 <= rowspan <= 1000 or not 1 <= colspan <= 1000:
                        raise ValueError()
                except ValueError:
                    self.omit(c, "invalid_table_span_assumed_one")
                    rowspan = colspan = 1
                val = clean(self.inline(c))
                cells.append(dict(row=ri, column=col, rowspan=rowspan, colspan=colspan,
                                  header=name(c) == "th", text=val, xml_path=self.paths[id(c)]))
                for ci in range(col, col + colspan):
                    occupied[ci] = ri + rowspan - 1
                col += colspan
            if cells:
                # Delimiters and coordinates are explicit additions, not source prose.
                val = f"row {ri + 1}\t" + "\t".join(f"col {c['column'] + 1}" +
                      (f"[rowspan={c['rowspan']},colspan={c['colspan']}]" if c['rowspan'] > 1 or c['colspan'] > 1 else "") +
                      "=" + c["text"] for c in cells)
                self.emit(val, row, "table_row", section, table_group=group, table_cells=cells,
                          rendering_addition="explicit_1_based_row_column_labels_and_tab_cell_delimiters")

    def walk(self, e, section):
        tag = name(e)
        if tag in EXCLUDE | MEDIA:
            self.inline(e)
        elif tag == "sec":
            title = child(e, "title")
            heading = clean(self.inline(title)) if title is not None else ""
            kind = e.get("sec-type") or (title_normalize(heading)[:80] if heading else "untitled_section")
            sub = self.section(e, section["section_id"], kind, heading)
            if title is not None:
                self.emit(heading, title, "section_heading", sub)
            for c in e:
                if c is not title:
                    self.walk(c, sub)
        elif tag == "p":
            self.paragraph(e, section, "front_abstract_paragraph" if section["section_type"] == "front_abstract" else "paragraph")
        elif tag in CONTAINERS:
            before = len(self.blocks)
            if clean(e.text or ""):
                self.emit(clean(e.text), e, tag + "_text", section)
            for c in e:
                self.walk(c, section)
                if clean(c.tail or ""):
                    self.emit(clean(c.tail), e, tag + "_text", section)
            if tag in {"list-item", "statement", "table-wrap-foot"}:
                for b in self.blocks[before:]:
                    if b["kind"] == "paragraph":
                        b["kind"] = {"list-item": "list_item_paragraph", "statement": "statement_paragraph", "table-wrap-foot": "table_note"}[tag]
        elif tag in {"title", "label", "term", "attrib"}:
            self.emit(clean(self.inline(e)), e, tag, section)
        elif tag in {"disp-formula", "inline-formula"}:
            self.emit(self.formula(e), e, "formula", section)
        elif tag == "math":
            self.emit(self.inline(e), e, "formula", section)
        elif tag == "caption":
            before = len(self.blocks)
            if clean(e.text or ""):
                self.emit(clean(e.text), e, "caption", section)
            for c in e:
                self.walk(c, section)
                if clean(c.tail or ""):
                    self.emit(clean(c.tail), e, "caption", section)
            for b in self.blocks[before:]:
                b["kind"] = "caption"
        elif tag in {"fig", "table-wrap"}:
            for c in e:
                if name(c) == "caption":
                    before = len(self.blocks)
                    if clean(c.text or ""):
                        self.emit(clean(c.text), c, tag + "_caption", section)
                    for cc in c:
                        self.walk(cc, section)
                        if clean(cc.tail or ""):
                            self.emit(clean(cc.tail), c, tag + "_caption", section)
                    for b in self.blocks[before:]:
                        b["kind"] = tag + "_caption"
                elif name(c) == "alternatives":
                    tables = children(c, "table")
                    if tables:
                        self.table(tables[0], section)
                    for alt in c:
                        if not tables or alt is not tables[0]:
                            self.omit(alt, "figure_table_alternative_omitted")
                elif name(c) == "table":
                    self.table(c, section)
                elif name(c) in {"label", "table-wrap-foot"} | MEDIA:
                    self.walk(c, section)
                else:
                    self.omit(c, "figure_table_noncaption_content_omitted")
        elif tag == "table":
            self.table(e, section)
        else:
            self.omit(e, "unsupported_block_element")

    def extract(self, meta):
        for abstract in children(meta, "abstract"):
            s = self.section(abstract, None, "front_abstract", "Local JATS abstract")
            if clean(abstract.text or ""):
                self.emit(clean(abstract.text), abstract, "front_abstract_paragraph", s)
            for c in abstract:
                self.walk(c, s)
        body = child(self.article, "body")
        if body is not None:
            s = self.section(body, None, "body", "Body")
            if clean(body.text or ""):
                self.emit(clean(body.text), body, "body_text", s)
            for c in body:
                self.walk(c, s)
        for c in self.article:
            if name(c) not in {"front", "body"}:
                self.omit(c, "outside_own_front_abstract_and_body")
        source_paragraphs = {self.paths[id(e)] for area in ([body] if body is not None else []) + children(meta, "abstract")
                             for e in area.iter() if name(e) == "p"}
        omitted = set()
        for path in source_paragraphs - self.handled_paragraphs:
            if any(path == o["xml_path"] or path.startswith(o["xml_path"] + "/") for o in self.omissions):
                omitted.add(path)
        unexplained = source_paragraphs - self.handled_paragraphs - omitted
        duplicates = sorted(p for p, n in self.paragraph_visits.items() if n > 1)
        coverage = dict(source_paragraphs=len(source_paragraphs), handled_paragraphs=len(source_paragraphs & self.handled_paragraphs),
                        handled_with_text=len(source_paragraphs & self.paragraphs_with_text), explicitly_omitted_paragraphs=len(omitted),
                        unexplained_paragraphs=sorted(unexplained),
                        duplicate_paragraph_visits=duplicates,
                        policy="every own front-abstract/body p handled once by structural walk or under an explicit omission; not a semantic completeness claim")
        return coverage


def extract_jats(data, pmid, pmcid, authoritative_title):
    digest = hashlib.sha256(data).hexdigest()
    root, article, meta, wrapper = article_root(data)
    ids = children(meta, "article-id")
    pmids = {own_text(e) for e in ids if e.get("pub-id-type") == "pmid"}
    pmcs = {own_text(e) for e in ids if e.get("pub-id-type") in {"pmc", "pmcid"}}
    pmcs = {"PMC" + p if p.isdigit() else p for p in pmcs}
    if not pmcid or pmids != {str(pmid)} or pmcs != {pmcid}:
        raise ValueError("own front PMID/PMCID mismatch or conflicting/missing IDs")
    groups = children(meta, "title-group")
    if len(groups) != 1 or len(children(groups[0], "article-title")) != 1:
        raise ValueError("missing or ambiguous own article title")
    title = ": ".join(own_text(e) for e in list(groups[0]) if name(e) in {"article-title", "subtitle"})
    comparison = compare_titles(authoritative_title, title)
    result = dict(source_sha256=digest, pmid=str(pmid), pmcid=pmcid, source_root=wrapper,
                  selected_root="article", title_comparison=comparison, license=permissions(meta),
                  source_article_type=article.get("article-type"),
                  status="major_title_mismatch_quarantined", blocks=[], sections=[], omissions=[], coverage={})
    if not comparison["accepted"]:
        return result
    ex = Extractor(root, article, digest)
    coverage = ex.extract(meta)
    if coverage["unexplained_paragraphs"] or coverage["duplicate_paragraph_visits"]:
        raise ValueError("unexplained source paragraph coverage gap")
    result.update(status="extracted_source_identity_validated", blocks=ex.blocks, sections=ex.sections,
                  omissions=ex.omissions, coverage=coverage)
    return result


def assemble_blocks(blocks):
    """Join rendered blocks, assigning document-relative codepoint offsets."""
    offset, parts = 0, []
    for b in blocks:
        if parts:
            offset += 2
        b["document_start"] = offset
        b["document_end"] = offset + len(b["text"])
        offset = b["document_end"]
        parts.append(b["text"])
    return "\n\n".join(parts)


def chunk_blocks(doc_id, blocks, target=4096, overlap=256):
    if not 0 <= overlap < target // 2 or target < 32:
        raise ValueError("invalid chunk bounds")
    document = assemble_blocks(blocks)
    ends = [b["document_end"] for b in blocks]
    chunks, start = [], 0
    while start < len(document):
        hard_end = min(start + target, len(document))
        end = hard_end
        if hard_end < len(document):
            boundaries = [p for p in ends if start + target // 2 <= p <= hard_end]
            if boundaries:
                end = max(boundaries)
            else:
                space = document.rfind(" ", max(start + 1, hard_end - 256), hard_end)
                if space > start:
                    end = space + 1
        spans = []
        for b in blocks:
            lo, hi = max(start, b["document_start"]), min(end, b["document_end"])
            if lo < hi:
                spans.append(dict(block_id=b["block_id"], chunk_start=lo-start, chunk_end=hi-start,
                                  block_start=lo-b["document_start"], block_end=hi-b["document_start"],
                                  document_start=lo, document_end=hi,
                                  xml_path=b["xml_path"], source_sha256=b["source_sha256"],
                                  section_id=b["section_id"]))
        chunks.append(dict(chunk_id=f"{doc_id}:chunk:{len(chunks):05d}", doc_id=doc_id, article_id=doc_id,
                           corpus_id=CORPUS_ID, text=document[start:end], document_start=start, document_end=end,
                           source_spans=spans, token_proxy=len(re.findall(r"[a-z0-9]+", document[start:end].lower()))))
        if end == len(document):
            break
        next_start = max(start + 1, end - overlap)
        # Prefer starting at a block boundary while keeping overlap <=256.
        boundaries = [b["document_start"] for b in blocks if next_start <= b["document_start"] < end]
        start = min(boundaries) if boundaries else next_start
    return document, chunks
