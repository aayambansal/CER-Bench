"""Offline identity primitives. Bare numbers require a namespace, never a guess."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "dev", "test")


def normalize_id(value, namespace=None):
    if value is None or isinstance(value, bool):
        raise ValueError(f"Invalid identity: {value!r}")
    text = str(value).strip()
    patterns = (
        ("PMID", r"(?:PMID\s*:\s*|https?://pubmed\.ncbi\.nlm\.nih\.gov/)([0-9]+)/?"),
        ("PMCID", r"(?:PMCID\s*:\s*)?PMC\s*([0-9]+)"),
        ("PMCID", r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/PMC([0-9]+)/?"),
        ("PMCID", r"https?://pmc\.ncbi\.nlm\.nih\.gov/articles/PMC([0-9]+)/?"),
    )
    for ns, pattern in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if match:
            if namespace and namespace.upper() != ns:
                raise ValueError("Identity namespace mismatch")
            number = int(match[1])
            break
    else:
        ns = namespace.upper() if namespace else None
        if ns not in {"PMID", "PMCID"} or not re.fullmatch(r"[0-9]+", text):
            raise ValueError(f"Unqualified or malformed identity: {value!r}")
        number = int(text)
    if number <= 0:
        raise ValueError("Identity must be positive")
    return f"PMID:{number}" if ns == "PMID" else f"PMC{number}"


def alias_key(value):
    text = str(value).strip()
    try:
        return normalize_id(text)
    except ValueError:
        return str(int(text)) if re.fullmatch(r"[0-9]+", text) else text


def identity_index(rows):
    aliases = defaultdict(set)
    for row in rows:
        canonical = normalize_id(row["pmid"], "PMID")
        values = [row.get("doc_id"), canonical, canonical.split(":")[1]]
        if row.get("pmcid"):
            pmcid = normalize_id(row["pmcid"], "PMCID")
            values += [pmcid, pmcid[3:]]
        for value in values:
            if value is not None and str(value).strip():
                aliases[alias_key(value)].add(canonical)
    return dict(aliases)


def resolve(value, aliases):
    candidates = aliases.get(alias_key(value), set())
    if len(candidates) != 1:
        raise ValueError(f"Unknown or ambiguous identity {value!r}: {sorted(candidates)}")
    return next(iter(candidates))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unique(rows, key):
    seen = set()
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in seen:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        seen.add(value)


def transform_refs(value, doc_fn=lambda x: x, chunk_fn=lambda x: x):
    """Walk annotated IDs (including passages and qrels); leave prose unchanged."""
    if isinstance(value, list):
        return [transform_refs(v, doc_fn, chunk_fn) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "doc_id" or key.endswith("_doc_id"):
            result[key] = doc_fn(item)
        elif key == "doc_ids" or key.endswith("_doc_ids"):
            result[key] = [doc_fn(v) for v in item]
        elif key == "chunk_id" or key.endswith("_chunk_id"):
            result[key] = chunk_fn(item)
        elif key == "chunk_ids" or key.endswith("_chunk_ids"):
            result[key] = [chunk_fn(v) for v in item]
        elif key == "qrels" and isinstance(item, dict):
            result[key] = {}
            for old, grade in item.items():
                new = doc_fn(old)
                if new in result[key]:
                    raise ValueError("Qrels aliases collapse; manual adjudication required")
                result[key][new] = grade
        else:
            result[key] = transform_refs(item, doc_fn, chunk_fn)
    return result


def annotated_ids(task):
    docs, chunks = set(), set()
    transform_refs(task, lambda v: docs.add(alias_key(v)) or v,
                   lambda v: chunks.add(str(v)) or v)
    return docs, chunks


def publish_directory(output, build):
    """Stage/validate before atomic rename, serializing cooperating writers."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = None
    try:
        os.close(fd)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        result = build(stage)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        stage.rename(output)
        return result
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        lock.unlink()
