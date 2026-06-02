"""v2.5: build the MIXED single-hop + multi-hop + unanswerable workload.

v2.5 puts an adaptive cost-escalation LADDER on a MIXED workload so each tier
earns its keep (cheap single-hop lookups answer immediately; hard multi-hop
escalate to decompose). MuSiQue already carries everything we need in the SAME
corpus: per-hop sub-questions give us single-hop queries, the full questions are
the multi-hop set, and native unanswerables drive STOP. No new corpus, no new
embeddings - just a new question population over data/questions.jsonl.

Single-hop quality: MuSiQue hop questions are often
decontextualized ("#1", "#2" placeholders that refer to a prior hop's answer) or
written as relation triples ("X >> rel"). Those are NOT standalone single-hop
queries. We keep only NATURAL-LANGUAGE hops with their own gold paragraph, and
take the lowest-hop-index clean hop per source (one single-hop query per source).

Eval integrity (the careful part):
  - GROUP-BY-SOURCE: a single-hop query INHERITS its source question's split, so a
    question's hops never cross calibration/validation/test (no hop-answer leak).
    Satisfied for free by construction; asserted anyway.
  - DISJOINT single/multi sources within a split: a source used for its single-hop
    query is never also used as a multi-hop item (no double-counting, no seeing the
    hop alone AND the full question in the same population).
  - The selected population is FROZEN in artifacts/mixed_manifest.json (seed 5252,
    deterministic). The test slice is touched once for the headline.

Outputs:
  data/questions_mixed.jsonl   - every original record tagged `query_type`, PLUS a
                                 derived single-hop record per single-hop-capable source
  artifacts/mixed_manifest.json - the frozen eval population (per split: single / multi
                                 / unanswerable ids) + counts

Usage:
  python scripts/prepare_mixed.py            # build + integrity asserts
  python scripts/prepare_mixed.py --dry-run  # report yield/feasibility, write nothing
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import config  # noqa: E402

SEED = 5252  # frozen selection seed for the mixed eval population

# Selection targets per split: ~2/3 single, ~1/3 multi among answerables;
# unanswerable = ALL (the native MuSiQue unanswerables, unchanged).
TARGETS = {
    "calibration": {"single": 100, "multi": 50},
    "validation": {"single": 100, "multi": 50},
    "test": {"single": 120, "multi": 60},
}
SPLIT_ORDER = ["calibration", "validation", "test"]  # fixed order -> deterministic rng

_WH = {"who", "what", "when", "where", "which", "whom", "whose", "why", "how"}

# Third-person personal pronouns. In MuSiQue hop questions these almost always
# co-refer to a PRIOR hop's answer ("Where did he die?", "Where did they migrate
# from?") - so a hop carrying one is only standalone-answerable if it ALSO names
# the referent ("Where did Andre Bloc live when he died?"). Hand-verified on the
# 34 hops that match: this guard keeps every named-referent hop and drops the
# dangling ones. (Found by sampling; ambiguous hops are dropped by hand.)
_PERSONAL_PRONOUNS = {
    "he", "she", "they", "him", "her", "them", "his", "their", "hers", "theirs",
    "himself", "herself", "themselves",
}


def _names_an_entity(question: str) -> bool:
    """Cheap proper-noun check: a capitalized word after the sentence-initial token
    (which is usually the wh-word). Robust enough to tell 'Where did he die?' (none)
    from 'Where did Andre Bloc live when he died?' (Bloc)."""
    words = question.split()
    return any(re.match(r"[A-Z][a-z]", w) for w in words[1:])


def is_natural_hop(question: str) -> bool:
    """Natural-language, standalone-answerable hop filter:
    has text, NO `#N` placeholder, NO `>>` relation triple, contains a wh-word OR
    ends with `?`, AND - if it carries a third-person personal pronoun - it must also
    name the referent (else the pronoun dangles to a prior hop and the query is not
    answerable alone)."""
    q = (question or "").strip()
    if not q:
        return False
    if re.search(r"#\d", q):
        return False
    if ">>" in q:
        return False
    toks = set(re.findall(r"[a-z']+", q.lower()))
    if not (q.endswith("?") or (_WH & toks)):
        return False
    if (_PERSONAL_PRONOUNS & toks) and not _names_an_entity(q):
        return False  # dangling coreference -> not standalone
    return True


def best_single_hop(rec: dict) -> dict | None:
    """The lowest-hop-index clean natural hop (with its own gold) for a source, or None
    if the source has no standalone-answerable hop."""
    cands = [
        h
        for h in rec.get("hops", [])
        if h.get("gold_doc_id") and is_natural_hop(h.get("question"))
    ]
    if not cands:
        return None
    return min(cands, key=lambda h: h["hop_index"])


def make_single_record(rec: dict, hop: dict) -> dict:
    """Derive a standalone single-hop query record from one hop of a source question.
    Inherits the source split (group-by-source). hop_index is reset to 0 inside the
    derived record (it is a one-hop question now); the id keeps the original hop index
    so it is traceable back to the source."""
    return {
        "id": f'{rec["id"]}__h{hop["hop_index"]}',
        "split": rec["split"],
        "answerable": True,
        "question": hop["question"],
        "answer": hop["answer"],
        "answer_aliases": [],
        "gold_doc_ids": [hop["gold_doc_id"]],
        "hops": [
            {
                "hop_index": 0,
                "question": hop["question"],
                "answer": hop["answer"],
                "gold_doc_id": hop["gold_doc_id"],
            }
        ],
        "n_hops": 1,
        "query_type": "single_hop",
        "source_id": rec["id"],
    }


def tag_query_type(rec: dict) -> str:
    """multi_hop for answerable items (all carry hops); unanswerable otherwise."""
    return "multi_hop" if rec.get("answerable") else "unanswerable"


def _norm_q(q: str) -> str:
    return " ".join((q or "").split())


def load_quality() -> dict:
    """Optional per-text standalone-answerability verdicts from audit_single_hop.py.
    Empty dict if the audit has not been run yet (the first prepare_mixed pass)."""
    path = config.ARTIFACTS_DIR / "single_hop_quality.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("verdicts_by_text", {})


def usable_single_sources(single_by_source: dict, quality: dict) -> tuple[set, dict]:
    """Clean the derived single-hop pool before selection:
      - DEDUP by normalized question text (keep the lowest source id) so no question
        text is counted twice across the whole population;
      - AND-GATE drop: exclude a question only when the LLM judged it not-standalone
        AND the heuristic finds no capitalized proper-noun anchor (so obscure named
        entities the LLM does not recognize are protected -> no famous-entity bias).
    Returns (usable_source_ids, stats)."""
    canonical: dict[str, str] = {}  # text -> lowest source id
    for src_id in sorted(single_by_source):
        canonical.setdefault(_norm_q(single_by_source[src_id]["question"]), src_id)

    usable, n_dup, n_andgate = set(), 0, 0
    for src_id, rec in single_by_source.items():
        text = _norm_q(rec["question"])
        if canonical[text] != src_id:
            n_dup += 1
            continue
        v = quality.get(text)
        if v is not None and not v.get("standalone", True) and not _names_an_entity(rec["question"]):
            n_andgate += 1
            continue
        usable.add(src_id)
    stats = {"dropped_duplicate_text": n_dup, "dropped_andgate": n_andgate, "usable": len(usable)}
    return usable, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report yield only; write nothing")
    args = ap.parse_args()

    src_rows = [json.loads(line) for line in (config.DATA_DIR / "questions.jsonl").open()]

    # --- build questions_mixed.jsonl rows ------------------------------------
    # 1) every original record, tagged with query_type (kept verbatim otherwise)
    # 2) a derived single-hop record for every single-hop-capable answerable source
    mixed_rows: list[dict] = []
    single_by_source: dict[str, dict] = {}  # source id -> derived single-hop record
    for rec in src_rows:
        out = dict(rec)
        out["query_type"] = tag_query_type(rec)
        mixed_rows.append(out)
        if rec.get("answerable"):
            hop = best_single_hop(rec)
            if hop is not None:
                single = make_single_record(rec, hop)
                single_by_source[rec["id"]] = single
                mixed_rows.append(single)

    # --- single-hop quality pass: dedup + AND-gate (see audit_single_hop.py) ---
    quality = load_quality()
    usable, qstats = usable_single_sources(single_by_source, quality)
    # tag the derived single-hop records with the LLM verdict + whether they survived
    for r in mixed_rows:
        if r.get("query_type") == "single_hop":
            v = quality.get(_norm_q(r["question"]))
            if v is not None:
                r["standalone_llm"] = bool(v.get("standalone", True))
            r["single_usable"] = r.get("source_id") in usable

    mixed_ids = {r["id"] for r in mixed_rows}

    # --- frozen selection (seed 5252; single & multi from DISJOINT sources) ---
    rng = random.Random(SEED)
    splits_sel: dict[str, dict] = {}
    counts: dict[str, dict] = {}
    capacity_ok = True
    for split in SPLIT_ORDER:
        answerable = sorted(
            (r for r in src_rows if r.get("split") == split and r.get("answerable")),
            key=lambda r: r["id"],
        )
        unanswerable = sorted(
            r["id"] for r in src_rows if r.get("split") == split and not r.get("answerable")
        )
        single_capable = [r for r in answerable if r["id"] in usable]

        n_single = TARGETS[split]["single"]
        n_multi = TARGETS[split]["multi"]
        if len(single_capable) < n_single:
            capacity_ok = False
            print(f"!! {split}: only {len(single_capable)} single-capable sources < target {n_single}")

        # pick single sources, then multi from the REMAINING answerable (disjoint)
        cap_shuffled = single_capable[:]
        rng.shuffle(cap_shuffled)
        single_sources = cap_shuffled[:n_single]
        single_source_ids = {r["id"] for r in single_sources}

        remaining = [r for r in answerable if r["id"] not in single_source_ids]
        if len(remaining) < n_multi:
            capacity_ok = False
            print(f"!! {split}: only {len(remaining)} remaining answerable < multi target {n_multi}")
        rng.shuffle(remaining)
        multi_sources = remaining[:n_multi]

        single_ids = sorted(single_by_source[r["id"]]["id"] for r in single_sources)
        multi_ids = sorted(r["id"] for r in multi_sources)

        splits_sel[split] = {
            "single": single_ids,
            "multi": multi_ids,
            "unanswerable": unanswerable,
        }
        counts[split] = {
            "single": len(single_ids),
            "multi": len(multi_ids),
            "unanswerable": len(unanswerable),
            "single_capable_available": len(single_capable),
            "answerable_available": len(answerable),
        }

    # --- integrity assertions -------------------------------------------------
    for split, sel in splits_sel.items():
        # (a) single and multi drawn from DISJOINT sources
        single_src = {sid.rsplit("__h", 1)[0] for sid in sel["single"]}
        multi_src = set(sel["multi"])  # multi ids ARE source ids
        overlap = single_src & multi_src
        assert not overlap, f"{split}: single/multi share {len(overlap)} sources: {sorted(overlap)[:5]}"

        # (b) every selected id resolves in questions_mixed.jsonl
        for kind in ("single", "multi", "unanswerable"):
            missing = [i for i in sel[kind] if i not in mixed_ids]
            assert not missing, f"{split}/{kind}: {len(missing)} ids missing from mixed: {missing[:5]}"

        # (c) group-by-source: each single-hop record sits in its source's split
        for sid in sel["single"]:
            src_split = single_by_source[sid.rsplit("__h", 1)[0]]["split"]
            assert src_split == split, f"{sid}: single split {src_split} != {split} (hop crossed splits)"

    # (d) global: no derived single-hop id collides with a source id
    src_ids = {r["id"] for r in src_rows}
    collisions = [s["id"] for s in single_by_source.values() if s["id"] in src_ids]
    assert not collisions, f"derived single ids collide with source ids: {collisions[:5]}"

    manifest = {
        "seed": SEED,
        "frozen": True,
        "note": "Frozen v2.5 eval population. Single-hop derived from clean natural hops "
        "(lowest hop index per source); single & multi from disjoint sources; single-hop "
        "inherits source split (group-by-source). Test slice touched once for the headline.",
        "ratios": {"single_frac_target": round(2 / 3, 4), "multi_frac_target": round(1 / 3, 4)},
        "targets": TARGETS,
        "single_hop_quality": {
            "derived_total": len(single_by_source),
            "audit_present": bool(quality),
            **qstats,
        },
        "counts": counts,
        "splits": splits_sel,
    }

    # --- report ---------------------------------------------------------------
    print("=== v2.5 mixed workload ===")
    print(f"source records: {len(src_rows)}  ->  mixed records: {len(mixed_rows)} "
          f"(+{len(single_by_source)} derived single-hop)")
    print(f"single-hop quality: audit_present={bool(quality)} | "
          f"dropped_duplicate_text={qstats['dropped_duplicate_text']} "
          f"dropped_andgate={qstats['dropped_andgate']} -> usable={qstats['usable']}")
    for split in SPLIT_ORDER:
        c = counts[split]
        ans = c["single"] + c["multi"]
        sfrac = c["single"] / ans if ans else 0
        print(f"  {split:12s} single={c['single']:3d} multi={c['multi']:3d} "
              f"unans={c['unanswerable']:3d}  | single frac of answerable = {sfrac:.2f} "
              f"(capacity: {c['single_capable_available']} single-capable / "
              f"{c['answerable_available']} answerable)")
    print(f"integrity: single/multi sources disjoint OK; ids resolve OK; group-by-source OK; "
          f"no id collisions OK  (capacity_ok={capacity_ok})")

    if args.dry_run:
        print("\n--dry-run: wrote nothing.")
        return 0 if capacity_ok else 1

    assert capacity_ok, "capacity shortfall (see !! lines above); not writing a partial population"

    out_q = config.DATA_DIR / "questions_mixed.jsonl"
    with out_q.open("w") as fh:
        for row in mixed_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_m = config.ARTIFACTS_DIR / "mixed_manifest.json"
    out_m.write_text(json.dumps(manifest, indent=2))

    qt = Counter(r["query_type"] for r in mixed_rows)
    print(f"\nwrote -> {out_q}  ({len(mixed_rows)} records; query_type {dict(qt)})")
    print(f"wrote -> {out_m}  (frozen population, seed {SEED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
