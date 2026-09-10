"""Offline, stdlib-only PubMed XML parsing with article-scoped identity.

No network, configuration, credentials, Bio, or yaml dependencies. Missing and
ambiguous records are diagnostics, never silently repaired from cited IDs.
"""

from collections import Counter
import re
import xml.etree.ElementTree as ET


def text(element):
    return "".join(element.itertext()).strip() if element is not None else ""


def local_tree(xml):
    """Accept namespaced PubMed/JATS while preserving caller's original bytes."""
    root = ET.fromstring(xml)
    for el in root.iter():
        el.tag = el.tag.rsplit("}", 1)[-1]
    return root


def normalize_pmc(value):
    value = str(value or "").strip().upper()
    if value.isdigit():
        value = "PMC" + value
    return value if re.fullmatch(r"PMC[0-9]+", value) else ""


def _unique(values, label, issues):
    values = sorted(set(v for v in values if v))
    if len(values) > 1:
        issues.append("conflicting_" + label)
        return ""
    return values[0] if values else ""


def parse_article(article):
    """Return (record, issues); only direct own ArticleIdList is authoritative."""
    issues = []
    medline = article.find("MedlineCitation")
    if medline is None:
        return None, ["missing_medline_citation"]
    pmid = _unique([text(e) for e in medline.findall("PMID")], "pmid", issues)
    if not pmid.isdigit():
        return None, issues + ["missing_or_invalid_pmid"]
    art = medline.find("Article")
    if art is None:
        return None, issues + ["missing_article"]
    ids = article.findall("PubmedData/ArticleIdList/ArticleId")
    own = {}
    for kind in ("pubmed", "pmc", "doi"):
        vals = [text(e) for e in ids if e.get("IdType") == kind]
        if kind == "pmc":
            if any(v and not normalize_pmc(v) for v in vals):
                issues.append("invalid_pmc")
            vals = [normalize_pmc(v) for v in vals]
        own[kind] = _unique(vals, kind, issues)
    if own["pubmed"] and own["pubmed"] != pmid:
        issues.append("own_pubmed_pmid_mismatch")
    # ArticleIdList has priority. ELocationID is also own-article scoped.
    doi = own["doi"]
    doi_path = "PubmedData/ArticleIdList/ArticleId[@IdType='doi']"
    if not doi and "conflicting_doi" not in issues:
        doi = _unique([text(e) for e in art.findall("ELocationID")
                       if e.get("EIdType") == "doi" and e.get("ValidYN") != "N"],
                      "doi", issues)
        doi_path = "MedlineCitation/Article/ELocationID[@EIdType='doi']"
    parts = []
    for e in art.findall("Abstract/AbstractText"):
        val = text(e)
        if val:
            parts.append((e.get("Label", "") + ": " if e.get("Label") else "") + val)
    date = art.find("Journal/JournalIssue/PubDate")
    year_text = "" if date is None else text(date.find("Year")) or text(date.find("MedlineDate"))
    match = re.search(r"\b(1[5-9][0-9]{2}|20[0-9]{2})\b", year_text)
    authors = []
    for e in art.findall("AuthorList/Author"):
        last, first = text(e.find("LastName")), text(e.find("ForeName"))
        name = (last + (", " + first if first else "")) if last else text(e.find("CollectiveName"))
        if name:
            authors.append(name)
    mesh = []
    for e in medline.findall("MeshHeadingList/MeshHeading"):
        descriptor = text(e.find("DescriptorName"))
        if descriptor:
            mesh.append(descriptor)
            mesh.extend(descriptor + "/" + text(q) for q in e.findall("QualifierName") if text(q))
    record = dict(pmid=pmid, pmcid=own["pmc"], doi=doi,
                  title=text(art.find("ArticleTitle")), abstract=" ".join(parts),
                  year=int(match.group()) if match else None,
                  venue=text(art.find("Journal/Title")),
                  venue_abbrev=text(art.find("Journal/ISOAbbreviation")),
                  authors=authors, mesh_terms=mesh,
                  publication_types=[text(e) for e in art.findall("PublicationTypeList/PublicationType")],
                  keywords=[text(e) for e in medline.findall("KeywordList/Keyword")],
                  source="pubmed", record_type="PubmedArticle",
                  metadata_issues=issues,
                  field_paths={"pmid": "MedlineCitation/PMID",
                               "pmcid": "PubmedData/ArticleIdList/ArticleId[@IdType='pmc']",
                               "doi": doi_path,
                               "title": "MedlineCitation/Article/ArticleTitle (itertext)",
                               "abstract": "MedlineCitation/Article/Abstract/AbstractText (itertext)",
                               "year": "MedlineCitation/Article/Journal/JournalIssue/PubDate"})
    return record, issues


def parse_pubmed_xml(xml, requested_pmids=None):
    """Return records plus explicit missing/unsupported/duplicate diagnostics.

    Book records are observed, but not coerced to journal article metadata.
    Duplicate records (even identical) are removed from usable records.
    """
    root = local_tree(xml)
    if root.tag != "PubmedArticleSet":
        raise ValueError("expected PubmedArticleSet, got " + root.tag)
    records, problems, returned = [], [], []
    for index, node in enumerate(root):
        pmid_path = "BookDocument/PMID" if node.tag == "PubmedBookArticle" else "MedlineCitation/PMID"
        pmid = text(node.find(pmid_path))
        if pmid:
            returned.append(pmid)
        if node.tag != "PubmedArticle":
            problems.append(dict(pmid=pmid or None, index=index, reason="unsupported_record_type", record_type=node.tag))
            continue
        record, issues = parse_article(node)
        if issues or record is None:
            problems.append(dict(pmid=pmid or None, index=index, reason="invalid_record", issues=issues))
        else:
            records.append(record)
    duplicates = sorted(k for k, v in Counter(returned).items() if v > 1)
    records = [r for r in records if r["pmid"] not in duplicates]
    requested = None if requested_pmids is None else [str(p) for p in requested_pmids]
    return dict(records=records, returned_pmids=returned, problems=problems,
                duplicate_pmids=duplicates,
                duplicate_requested_pmids=[] if requested is None else sorted(k for k, v in Counter(requested).items() if v > 1),
                missing_pmids=[] if requested is None else sorted(set(requested) - set(returned)),
                unexpected_pmids=[] if requested is None else sorted(set(returned) - set(requested)))


def parse_records(xml):
    """Compatibility list API, fail closed instead of silently dropping records."""
    result = parse_pubmed_xml(xml)
    if result["problems"] or result["duplicate_pmids"]:
        raise ValueError("PubMed XML contains unsupported, invalid, or duplicate records")
    return result["records"]


def inspect_fulltext(xml):
    """Raw JATS front identities and permissions only; no body text restored.

    Identity never comes from filename, back references, subarticles or body.
    Original license XML retains namespaces; status is not a rights grant.
    """
    original = ET.fromstring(xml)
    root = local_tree(xml)
    articles = [root] if root.tag == "article" else root.findall("article") if root.tag == "pmc-articleset" else []
    result = dict(status="unsupported_or_multiple_articles", pmids=[], pmcids=[],
                  license_xml=[], namespace=original.tag.split("}")[0][1:] if "}" in original.tag else "",
                  redistribution_status="not_assessed_no_rights_granted")
    if len(articles) != 1:
        return result
    article = articles[0]
    ids = article.findall("front/article-meta/article-id")
    pmids = [text(e) for e in ids if e.get("pub-id-type") == "pmid"]
    pmcs = [text(e) for e in ids if e.get("pub-id-type") in ("pmc", "pmcid")]
    result.update(pmids=sorted(set(pmids)), pmcids=sorted(set(normalize_pmc(v) for v in pmcs)))
    def children(e, name):
        return [c for c in e if c.tag.rsplit("}", 1)[-1] == name]
    originals = [original] if original.tag.rsplit("}", 1)[-1] == "article" else children(original, "article")
    for front in children(originals[0], "front"):
        for meta in children(front, "article-meta"):
            for permission in children(meta, "permissions"):
                result["license_xml"].append(ET.tostring(permission, encoding="unicode"))
    result["status"] = "front_identity_candidate" if (len(result["pmids"]) == len(result["pmcids"]) == 1
                          and result["pmids"][0].isdigit() and result["pmcids"][0]) else "missing_or_conflicting_front_ids"
    return result
