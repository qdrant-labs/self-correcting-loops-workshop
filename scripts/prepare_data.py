"""Phase 2: build the MuSiQue workshop dataset.

MuSiQue-Full gives us both archetypes in one config: answerable items carry
per-hop gold supporting paragraphs; unanswerable items have no gold and drive the
"stop" archetype. This script:

  - selects a question subset (answerable + unanswerable + a hero pool)
  - builds a deduplicated paragraph corpus (union over all selected items)
  - resolves per-hop gold to stable, content-addressed doc ids
  - partitions non-hero questions into disjoint calibration / validation / test
    splits (seeded, stratified by answerable x hop-count)
  - writes data/corpus.jsonl, data/questions.jsonl, data/dataset_meta.json
  - runs integrity assertions: splits disjoint; every gold paragraph reachable

Eval integrity: MuSiQue is locked as primary BEFORE any tuning;
this script does no model tuning. The three splits have distinct jobs
(calibration = thresholds, validation = signal/policy selection, test = final
lift, touched once). Hero queries are illustrative demos, excluded from all three
splits so a demo never touches the aggregate claim.

Usage:
  python scripts/prepare_data.py --smoke          # tiny, validates the pipeline
  python scripts/prepare_data.py                  # full build (default sizes)
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import config  # noqa: E402

# Fixed namespace so a paragraph's doc id is a pure function of its (title, text):
# same passage -> same id across rebuilds, and dedup falls out for free.
CORPUS_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "musique.qdrant.workshop.corpus")


# --- small casting / normalization helpers -----------------------------------
def to_bool(x) -> bool:
    return x in (True, "True", "true", 1, "1")


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def norm(s: str) -> str:
    return " ".join((s or "").split())


def doc_id_for(title: str, text: str) -> str:
    return str(uuid.uuid5(CORPUS_NS, norm(title) + "␟" + norm(text)))


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- normalize one raw MuSiQue record ----------------------------------------
def normalize_item(raw: dict) -> dict:
    paras = []
    for pos, p in enumerate(raw.get("paragraphs") or []):
        paras.append(
            {
                "pos": pos,
                "idx": to_int(p.get("idx")),
                "title": p.get("title") or "",
                "text": p.get("paragraph_text") or "",
                "is_supporting": to_bool(p.get("is_supporting")),
            }
        )
    hops = []
    for hop_index, h in enumerate(raw.get("question_decomposition") or []):
        hops.append(
            {
                "hop_index": hop_index,
                "question": h.get("question") or "",
                "answer": h.get("answer") or "",
                "support_pos": to_int(h.get("paragraph_support_idx")),
            }
        )
    aliases = raw.get("answer_aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return {
        "id": raw.get("id"),
        "question": raw.get("question") or "",
        "answer": raw.get("answer") or "",
        "answer_aliases": list(aliases),
        "answerable": to_bool(raw.get("answerable")),
        "n_hops": len(hops),
        "paragraphs": paras,
        "hops": hops,
    }


def build_question_record(item: dict) -> dict | None:
    """Resolve gold to doc ids. Returns None for answerable items that lack any
    supporting paragraph (a data quirk we skip rather than mis-label)."""
    pos_to_doc = {p["pos"]: doc_id_for(p["title"], p["text"]) for p in item["paragraphs"]}

    full_gold = sorted(
        {pos_to_doc[p["pos"]] for p in item["paragraphs"] if p["is_supporting"]}
    )
    hops_out = []
    for hop in item["hops"]:
        gold_doc = pos_to_doc.get(hop["support_pos"]) if hop["support_pos"] is not None else None
        hops_out.append(
            {
                "hop_index": hop["hop_index"],
                "question": hop["question"],
                "answer": hop["answer"],
                "gold_doc_id": gold_doc,
            }
        )

    if item["answerable"] and not full_gold:
        return None  # answerable but no gold annotation -> unusable, skip

    return {
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "answer_aliases": item["answer_aliases"],
        "answerable": item["answerable"],
        "n_hops": item["n_hops"],
        "hops": hops_out,
        "gold_doc_ids": full_gold,  # support for ALL hops (empty for unanswerable)
    }


def collect_corpus(items: list[dict], records_by_id: dict[str, dict]) -> dict[str, dict]:
    """Union of all paragraphs across selected items, deduped by content id.
    Each corpus doc records which (question, hop) it is gold for (per-hop gold in
    the payload, so a trace inspector can see a missing hop directly)."""
    corpus: dict[str, dict] = {}
    for item in items:
        rec = records_by_id.get(item["id"])
        # map this question's per-hop gold doc -> hop_index, for the supports tag
        hop_of_doc = {}
        if rec:
            for hop in rec["hops"]:
                if hop["gold_doc_id"]:
                    hop_of_doc.setdefault(hop["gold_doc_id"], hop["hop_index"])
        for p in item["paragraphs"]:
            did = doc_id_for(p["title"], p["text"])
            doc = corpus.get(did)
            if doc is None:
                doc = {"doc_id": did, "title": norm(p["title"]), "text": norm(p["text"]), "supports": []}
                corpus[did] = doc
            if p["is_supporting"] and rec is not None:
                doc["supports"].append(
                    {"question_id": item["id"], "hop_index": hop_of_doc.get(did)}
                )
    return corpus


def stratified_split(records: list[dict], rng: random.Random, props: dict[str, float]) -> None:
    """Assign each record a 'split' in place, stratified by (answerable, hop bucket)
    so every split holds answerable + unanswerable items across hop counts."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        strata[(r["answerable"], min(r["n_hops"], 4))].append(r)

    names = list(props.keys())
    for _, group in strata.items():
        rng.shuffle(group)
        n = len(group)
        # cumulative boundaries; remainder goes to the last split (test)
        cuts = []
        acc = 0.0
        for name in names[:-1]:
            acc += props[name]
            cuts.append(round(acc * n))
        bounds = [0, *cuts, n]
        for i, name in enumerate(names):
            for r in group[bounds[i] : bounds[i + 1]]:
                r["split"] = name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdsaglam/musique")
    ap.add_argument("--dataset-config", default="default")  # = MuSiQue-Full
    ap.add_argument("--source-split", default="train")
    ap.add_argument("--n-answerable", type=int, default=1600)
    ap.add_argument("--n-unanswerable", type=int, default=400)
    ap.add_argument("--n-hero-answerable", type=int, default=24)
    ap.add_argument("--n-hero-unanswerable", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny sizes to validate the pipeline")
    args = ap.parse_args()

    if args.smoke:
        args.n_answerable, args.n_unanswerable = 60, 20
        args.n_hero_answerable, args.n_hero_unanswerable = 6, 4

    import warnings

    warnings.filterwarnings("ignore")
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from datasets import load_dataset

    print(f"loading {args.dataset}:{args.dataset_config} [{args.source_split}] ...")
    raw = load_dataset(args.dataset, args.dataset_config, split=args.source_split)

    # normalize + build records; partition by answerability
    answerable_items, unanswerable_items = [], []
    records_by_id: dict[str, dict] = {}
    skipped_no_gold = 0
    seen_ids = set()
    for raw_item in raw:
        item = normalize_item(raw_item)
        if not item["id"] or item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        rec = build_question_record(item)
        if rec is None:
            skipped_no_gold += 1
            continue
        records_by_id[item["id"]] = rec
        (answerable_items if item["answerable"] else unanswerable_items).append(item)

    rng = random.Random(args.seed)
    rng.shuffle(answerable_items)
    rng.shuffle(unanswerable_items)

    # --- hero pool: spread answerable across hop counts, plus some unanswerable
    by_hops: dict[int, list[dict]] = defaultdict(list)
    for it in answerable_items:
        by_hops[min(it["n_hops"], 4)].append(it)
    hero_ans: list[dict] = []
    want_per_hop = max(1, args.n_hero_answerable // 3)
    for h in (4, 3, 2):  # bias the hero pool toward harder multi-hop
        take = want_per_hop if h != 2 else args.n_hero_answerable - len(hero_ans)
        hero_ans.extend(by_hops[h][:take])
    hero_ans = hero_ans[: args.n_hero_answerable]
    hero_unans = unanswerable_items[: args.n_hero_unanswerable]
    hero_ids = {it["id"] for it in hero_ans} | {it["id"] for it in hero_unans}

    # --- split pools: remaining items, excluding hero
    rest_ans = [it for it in answerable_items if it["id"] not in hero_ids][: args.n_answerable]
    rest_unans = [it for it in unanswerable_items if it["id"] not in hero_ids][: args.n_unanswerable]

    selected_items = hero_ans + hero_unans + rest_ans + rest_unans

    # --- corpus (union over ALL selected, incl. hero) ---------------------------
    corpus = collect_corpus(selected_items, records_by_id)
    corpus_ids = set(corpus.keys())

    # --- assemble question records with split labels ----------------------------
    split_records = [records_by_id[it["id"]] for it in (rest_ans + rest_unans)]
    stratified_split(split_records, rng, {"calibration": 0.30, "validation": 0.30, "test": 0.40})

    hero_records = []
    for it in hero_ans + hero_unans:
        r = dict(records_by_id[it["id"]])
        r["split"] = "hero"
        r["is_hero"] = True
        r["archetype"] = "stop" if not r["answerable"] else ("multihop" if r["n_hops"] >= 3 else "multihop_2")
        hero_records.append(r)

    all_question_records = split_records + hero_records

    # --- integrity assertions ----------------------------------------------------
    # (1) splits disjoint by id
    ids_by_split = defaultdict(set)
    for r in all_question_records:
        ids_by_split[r["split"]].add(r["id"])
    all_ids = [i for s in ids_by_split.values() for i in s]
    assert len(all_ids) == len(set(all_ids)), "split id collision: splits not disjoint"

    # (2) every answerable question's gold is reachable in the corpus
    unreachable = [
        r["id"]
        for r in all_question_records
        if r["answerable"] and not set(r["gold_doc_ids"]).issubset(corpus_ids)
    ]
    assert not unreachable, f"{len(unreachable)} questions have gold missing from corpus"

    # --- write outputs -----------------------------------------------------------
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_rows = sorted(corpus.values(), key=lambda d: d["doc_id"])
    write_jsonl(config.DATA_DIR / "corpus.jsonl", corpus_rows)
    write_jsonl(config.DATA_DIR / "questions.jsonl", split_records + hero_records)

    split_sizes = {s: len(ids) for s, ids in ids_by_split.items()}
    ans_by_split = {
        s: sum(1 for r in all_question_records if r["split"] == s and r["answerable"])
        for s in ids_by_split
    }
    meta = {
        "primary_dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "source_split": args.source_split,
        "note": "MuSiQue locked as primary before tuning. Splits drawn from MuSiQue train, seeded + disjoint; distinct jobs.",
        "seed": args.seed,
        "n_corpus_docs": len(corpus_rows),
        "n_questions_total": len(all_question_records),
        "split_sizes": split_sizes,
        "answerable_by_split": ans_by_split,
        "unanswerable_by_split": {s: split_sizes[s] - ans_by_split[s] for s in split_sizes},
        "hop_distribution": dict(sorted(Counter(r["n_hops"] for r in all_question_records).items())),
        "skipped_answerable_no_gold": skipped_no_gold,
    }
    (config.DATA_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== build summary ===")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote -> {config.DATA_DIR}/  (corpus.jsonl, questions.jsonl, dataset_meta.json)")
    print("integrity: splits disjoint OK; all answerable gold reachable OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
