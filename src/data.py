"""Data access layer for the workshop dataset.

Read-only over the files written by scripts/prepare_data.py + scripts/prepare_mixed.py:
  data/corpus.jsonl, data/questions.jsonl, data/questions_mixed.jsonl, data/dataset_meta.json

Also defines the single text representation used for indexing and (mirrored) for
querying: "title. text" - the Wikipedia title carries the entity name, which helps
both dense and lexical retrieval match.
"""
from __future__ import annotations

import json
from functools import lru_cache

import config


def doc_embed_text(doc: dict) -> str:
    """The text we embed / index for a corpus paragraph (title + body)."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title}. {text}" if title else text


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, dict]:
    """doc_id -> {doc_id, title, text, supports}. Cached (loaded once per process)."""
    path = config.DATA_DIR / "corpus.jsonl"
    out: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def load_questions(split: str | None = None) -> list[dict]:
    """All question records, optionally filtered to one split
    (calibration / validation / test / hero)."""
    path = config.DATA_DIR / "questions.jsonl"
    rows = [json.loads(line) for line in path.open()]
    if split is not None:
        rows = [r for r in rows if r.get("split") == split]
    return rows


def load_questions_mixed(split: str | None = None) -> list[dict]:
    """v2.5 mixed workload: every original record (tagged `query_type`) plus the
    derived single-hop records, from data/questions_mixed.jsonl. Optionally filter to
    one split. Built by scripts/prepare_mixed.py."""
    path = config.DATA_DIR / "questions_mixed.jsonl"
    rows = [json.loads(line) for line in path.open()]
    if split is not None:
        rows = [r for r in rows if r.get("split") == split]
    return rows


def load_mixed_manifest() -> dict:
    """The frozen v2.5 eval population (seed 5252): per-split single/multi/unanswerable
    ids + counts. See artifacts/mixed_manifest.json."""
    return json.loads((config.ARTIFACTS_DIR / "mixed_manifest.json").read_text())


def load_mixed_eval(split: str) -> list[dict]:
    """The selected v2.5 eval records for one split (single + multi + unanswerable),
    resolved from questions_mixed.jsonl in manifest order. This is the frozen
    population the ladder is evaluated on; the test split is touched once."""
    manifest = load_mixed_manifest()
    sel = manifest["splits"][split]
    by_id = {r["id"]: r for r in load_questions_mixed()}
    ids = [*sel["single"], *sel["multi"], *sel["unanswerable"]]
    return [by_id[i] for i in ids]


def load_meta() -> dict:
    return json.loads((config.DATA_DIR / "dataset_meta.json").read_text())
