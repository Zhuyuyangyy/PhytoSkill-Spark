"""Vendor a local, hashed copy of the TCM-Mind-RAG knowledge data.

Run manually when the corpus is refreshed:

    python -m corpus.vendor --source "D:/ZYY Project/TCM-Mind-RAG/backend/data"

The upstream project is never imported or modified. Only plain YAML/JSON data is
read, and the output is a deterministic ``index.json`` with per-chunk evidence
ids and line ranges. Re-running produces a byte-identical index for identical
input, which is what makes a citation reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

CORPUS_DIR = Path(__file__).resolve().parent
INDEX_PATH = CORPUS_DIR / "index.json"
CORPUS_VERSION = "tcm-mind-rag-data-2026-05-29"

# Every chunk is tagged with the species it can legitimately speak about. The
# corpus is TCM syndrome knowledge, so a plant-physiology query is out of scope
# and the retriever must be able to say so.
SPECIES_HINTS = {
    "黄芪": "黄芪", "人参": "人参", "当归": "当归", "甘草": "甘草", "白术": "白术",
    "茯苓": "茯苓", "川芎": "川芎", "金银花": "金银花", "枸杞": "枸杞",
}
PHENOTYPE_HINTS = {
    "黄化": "leaf_yellowing", "叶片黄化": "leaf_yellowing", "枯黄": "leaf_yellowing",
    "萎蔫": "wilting", "枯萎": "wilting", "斑点": "leaf_spot",
}


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _find_line(lines: list[str], needle: str, start: int = 0) -> int:
    """1-based line number of the first line containing ``needle``."""
    for index in range(start, len(lines)):
        if needle in lines[index]:
            return index + 1
    return start + 1


def _species_tags(text: str) -> list[str]:
    return [species for species in SPECIES_HINTS if species in text]


def _phenotype_tags(text: str) -> list[str]:
    tags = []
    for hint, phenotype in PHENOTYPE_HINTS.items():
        if hint in text and phenotype not in tags:
            tags.append(phenotype)
    return tags


def _stable_id(source_file: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{source_file}#{ordinal}".encode("utf-8")).hexdigest()[:12]
    return f"corpus-{digest}"


def build_chunks(source_dir: Path) -> list[dict]:
    """Turn each YAML/JSON record into one retrievable, locatable chunk."""
    chunks: list[dict] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".yaml", ".yml", ".json"):
            continue
        relative = path.name
        if path.suffix.lower() == ".json":
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeError):
                continue
            records = _json_records(document, path)
            for ordinal, (title, content, extra, line_start, line_end) in enumerate(records, start=1):
                chunks.append(_chunk(relative, "json_record", ordinal, title, content, extra,
                                     line_start, line_end))
            continue
        lines = _lines_of(path)
        document = _read_yaml(path)
        if not isinstance(document, dict):
            continue
        for key, records in document.items():
            if not isinstance(records, list):
                continue
            for ordinal, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    continue
                title = str(record.get("syndrome") or record.get("name")
                            or record.get("symptom") or f"{key}-{ordinal}")
                content = _describe(record)
                if not content.strip():
                    continue
                anchor = title if len(title) >= 2 else f"{key}:"
                line_start = _find_line(lines, anchor)
                line_end = min(len(lines), line_start + max(1, content.count("\n")))
                extra = {"record_kind": key}
                chunks.append(_chunk(relative, key, ordinal, title, content, extra,
                                     line_start, line_end))
    return chunks


def _json_records(document: Any, path: Path) -> list[tuple[str, str, dict, int, int]]:
    """Extract records with real line ranges by locating each anchor in the text.

    A JSON document is not line-oriented, so the anchor (the record's own title)
    is searched for in the raw text. When the anchor cannot be located, the
    range falls back to the whole file rather than to a fabricated line number.
    """
    lines = _lines_of(path)
    records: list[tuple[str, str, dict, int, int]] = []
    if isinstance(document, list):
        items = [(str(item.get("syndrome") or item.get("name") or item.get("id") or f"record-{index}"),
                  item, {"record_kind": "record"})
                 for index, item in enumerate(document, start=1) if isinstance(item, dict)]
    elif isinstance(document, dict):
        items = []
        for key, value in document.items():
            if isinstance(value, list):
                items.extend((str(item.get("syndrome") or item.get("name") or f"{key}-{index}"),
                              item, {"record_kind": key})
                             for index, item in enumerate(value, start=1) if isinstance(item, dict))
    else:
        return records
    cursor = 0
    for title, item, extra in items:
        content = _describe(item)
        if not content.strip():
            continue
        anchor = _json_anchor(title)
        line_start = _find_line(lines, anchor, cursor) if anchor else 0
        if not line_start:
            line_start, line_end = 1, max(1, len(lines))
        else:
            cursor = line_start
            line_end = min(len(lines), line_start + max(1, content.count("\n")))
        records.append((title, content, extra, line_start, line_end))
    return records


def _json_anchor(title: str) -> str:
    """A quoted JSON key/value fragment that uniquely opens one record."""
    if not title or len(title) < 2:
        return ""
    return f'"{title}"'


def _describe(record: dict) -> str:
    """Flatten a record into a readable passage without inventing content."""
    parts: list[str] = []
    for key, value in record.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            rendered = "、".join(str(item) for item in value)
        elif isinstance(value, dict):
            rendered = "；".join(f"{k}={v}" for k, v in value.items())
        else:
            rendered = str(value)
        parts.append(f"{key}: {rendered}")
    return "\n".join(parts)


def _chunk(source_file: str, source_kind: str, ordinal: int, title: str, content: str,
           extra: dict, line_start: int | None, line_end: int | None) -> dict:
    combined = f"{title}\n{content}"
    return {
        "evidence_id": _stable_id(source_file, ordinal),
        "source_file": source_file,
        "source_kind": source_kind,
        "title": title,
        "content": content,
        "line_start": line_start if line_start is not None else 1,
        "line_end": line_end if line_end is not None else 1,
        "species_tags": _species_tags(combined),
        "phenotype_tags": _phenotype_tags(combined),
        **extra,
    }


def vendor(source_dir: Path, output: Path = INDEX_PATH) -> dict:
    chunks = build_chunks(source_dir)
    index = {
        "corpus_version": CORPUS_VERSION,
        "source": {"kind": "tcm-mind-rag", "path": str(source_dir),
                   "note": "Vendored copy; upstream project is not imported or modified."},
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    payload = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    return {
        "corpus_version": CORPUS_VERSION,
        "chunks": len(chunks),
        "index": str(output),
        "index_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_files": sorted({chunk["source_file"] for chunk in chunks}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corpus.vendor", description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory of TCM-Mind-RAG YAML/JSON data to vendor")
    parser.add_argument("--output", type=Path, default=INDEX_PATH)
    args = parser.parse_args(argv)
    if not args.source.is_dir():
        print(f"source directory not found: {args.source}", file=sys.stderr)
        return 2
    print(json.dumps(vendor(args.source, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
