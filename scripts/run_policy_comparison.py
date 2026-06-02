"""v2.5 Step 4 (validation): the policy comparison + prune decision.

The v2.5 headline claim is that the adaptive LADDER beats any FIXED policy on the
cost/quality frontier of a MIXED workload. This script tests that on VALIDATION (the
prune split - the test slice is touched only by the final headline) using a
counterfactual-style design: for each query we precompute the retrieval under each
action (baseline / ColBERT / IRCoT) and score RETRIEVAL PRECISION (recall@1/@3, MRR,
full_gold@3 - the PRIMARY metric, LLM-free except IRCoT's sub-query generation, which is
cached). The ladder's tier decision is the deterministic signal gate (LLM-free), so the
whole comparison needs no answer generation - that lands in the test headline.

Policies compared:
  always_answer    - answer from the hybrid baseline top-3 (tier 1 only)
  always_colbert   - ColBERT late-interaction on every query (tier 2)
  always_decompose - IRCoT on every query (tier 3)
  ladder           - the adaptive gate: T1 if confident, else T3 if multi-hop-looking, else T2

Reports per policy, by query type: precision metrics + cost (mean retrieval calls + mean
LLM sub-query calls). Plus the ladder's tier fire rates and a prune verdict (does each
tier earn its keep?).

Usage:  python scripts/run_policy_comparison.py        (IRCoT cached to artifacts/ircot_mixed_val.json)
        python scripts/run_policy_comparison.py --recompute-ircot
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import agent  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import policy as policy_mod  # noqa: E402
import retrieval  # noqa: E402
import signals as sg  # noqa: E402

IRCOT_CACHE = config.ARTIFACTS_DIR / "ircot_mixed_val.json"


def precision_scores(doc_ids, gold) -> dict:
    """recall@1/@3, full_gold@3, MRR(first gold) for one query."""
    gold = list(gold or [])
    gs = set(gold)
    top1, top3 = set(doc_ids[:1]), set(doc_ids[:3])
    rank = next((i + 1 for i, d in enumerate(doc_ids) if d in gs), 0)
    return {
        "recall@1": len(top1 & gs) / len(gs) if gs else float("nan"),
        "recall@3": len(top3 & gs) / len(gs) if gs else float("nan"),
        "full_gold@3": 1.0 if gs and gs.issubset(top3) else 0.0,
        "mrr": (1.0 / rank) if rank else 0.0,
    }


def load_ircot_cache() -> dict:
    return json.loads(IRCOT_CACHE.read_text()) if IRCOT_CACHE.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-ircot", action="store_true")
    args = ap.parse_args()

    th = sg.load_thresholds(path=config.ARTIFACTS_DIR / "thresholds_mixed.json")
    detector = th.get("_detector", sg.DEFAULT_DETECTOR)
    val = [q for q in data.load_mixed_eval("validation") if q.get("answerable") and q.get("gold_doc_ids")]
    print(f"validation answerable: {len(val)} "
          f"(single={sum(q['query_type']=='single_hop' for q in val)}, "
          f"multi={sum(q['query_type']=='multi_hop' for q in val)})")
    print(f"ladder gate: {th['_weakness_signals']} | answer from top-{config.ANSWER_K}\n")

    ircot_cache = {} if args.recompute_ircot else load_ircot_cache()
    rows = []
    for i, q in enumerate(val):
        qid = q["id"]; gold = q["gold_doc_ids"]; question = q["question"]
        enc = retrieval.encode_query(question)
        base = retrieval.search(question, mode="hybrid", k=config.TOP_K, fusion="rrf", encoded=enc)
        col = retrieval.colbert_search(question, n_prefetch=config.RETRIEVE_N, k=config.TOP_K)
        if qid in ircot_cache:
            ir = ircot_cache[qid]
        else:
            res, _ = agent.ircot_search(question, k=config.TOP_K, fusion="rrf", encoded=enc)
            ir = {"doc_ids": res.doc_ids, "n_sub": len(res.sub_queries)}
            ircot_cache[qid] = ir
        # ladder routing decision from the baseline signals (LLM-free)
        report = sg.read_signals(base, th, detector=detector)
        multi = policy_mod.looks_multi_hop(question)
        tier = 1 if report.healthy else (3 if multi else 2)
        rows.append({
            "qid": qid, "qtype": q["query_type"], "tier": tier, "n_sub": ir["n_sub"],
            "always_answer": precision_scores(base.doc_ids, gold),
            "always_colbert": precision_scores(col.doc_ids, gold),
            "always_decompose": precision_scores(ir["doc_ids"], gold),
            "ladder": precision_scores({1: base.doc_ids, 2: col.doc_ids, 3: ir["doc_ids"]}[tier], gold),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(val)} ...")
    IRCOT_CACHE.write_text(json.dumps(ircot_cache, indent=2))

    # per-query cost (retrieval calls, LLM sub-query calls)
    def cost(policy, r):
        if policy == "always_decompose":
            return {"retr": 1 + r["n_sub"], "llm": r["n_sub"]}
        if policy == "ladder":
            return {"retr": (1 + r["n_sub"]) if r["tier"] == 3 else 1, "llm": r["n_sub"] if r["tier"] == 3 else 0}
        return {"retr": 1, "llm": 0}  # always_answer / always_colbert

    POLICIES = ["always_answer", "always_colbert", "always_decompose", "ladder"]
    METRICS = ["recall@1", "recall@3", "full_gold@3", "mrr"]

    def agg(rows_sub):
        out = {}
        for p in POLICIES:
            q = {m: round(st.mean(r[p][m] for r in rows_sub), 4) for m in METRICS}
            costs = [cost(p, r) for r in rows_sub]
            q["cost_retr"] = round(st.mean(c["retr"] for c in costs), 3)
            q["cost_llm"] = round(st.mean(c["llm"] for c in costs), 3)
            out[p] = q
        return out

    overall = agg(rows)
    by_type = {qt: agg([r for r in rows if r["qtype"] == qt]) for qt in ("single_hop", "multi_hop")}
    tier_dist = {qt: {t: sum(1 for r in rows if r["qtype"] == qt and r["tier"] == t) for t in (1, 2, 3)}
                 for qt in ("single_hop", "multi_hop")}

    # ---- report ----
    def table(title, a):
        print(f"\n=== {title} ===")
        print(f"{'policy':18s} {'rec@1':>6s} {'rec@3':>6s} {'fg@3':>6s} {'MRR':>6s} {'retr':>6s} {'llm':>6s}")
        for p in POLICIES:
            v = a[p]
            print(f"{p:18s} {v['recall@1']:6.3f} {v['recall@3']:6.3f} {v['full_gold@3']:6.3f} "
                  f"{v['mrr']:6.3f} {v['cost_retr']:6.2f} {v['cost_llm']:6.2f}")
    table("OVERALL (validation answerable)", overall)
    table("single-hop", by_type["single_hop"])
    table("multi-hop", by_type["multi_hop"])
    print(f"\nladder tier fire rates: single={tier_dist['single_hop']}  multi={tier_dist['multi_hop']}")

    # ---- prune verdict ----
    print("\n=== prune analysis ===")
    # T2 (ColBERT) earns its keep? compare always_colbert vs always_answer on recall@3 + MRR
    aa, ac, ad, la = overall["always_answer"], overall["always_colbert"], overall["always_decompose"], overall["ladder"]
    print(f"T2 ColBERT vs answer (overall): recall@3 {aa['recall@3']:.3f}->{ac['recall@3']:.3f} "
          f"({ac['recall@3']-aa['recall@3']:+.3f}), MRR {aa['mrr']:.3f}->{ac['mrr']:.3f} ({ac['mrr']-aa['mrr']:+.3f})")
    # on the queries the ladder ROUTES to T2, does colbert beat baseline?
    t2 = [r for r in rows if r["tier"] == 2]
    if t2:
        b3 = st.mean(r["always_answer"]["recall@3"] for r in t2); c3 = st.mean(r["always_colbert"]["recall@3"] for r in t2)
        bm = st.mean(r["always_answer"]["mrr"] for r in t2); cm = st.mean(r["always_colbert"]["mrr"] for r in t2)
        print(f"   on the {len(t2)} ladder-T2 queries: recall@3 {b3:.3f}->{c3:.3f} ({c3-b3:+.3f}), MRR {bm:.3f}->{cm:.3f} ({cm-bm:+.3f})")
    # ladder vs always_decompose: comparable quality at lower cost?
    print(f"ladder vs always_decompose: recall@3 {la['recall@3']:.3f} vs {ad['recall@3']:.3f} "
          f"({la['recall@3']-ad['recall@3']:+.3f}) at LLM cost {la['cost_llm']:.2f} vs {ad['cost_llm']:.2f}")
    print(f"ladder vs always_answer:    recall@3 {la['recall@3']:.3f} vs {aa['recall@3']:.3f} "
          f"({la['recall@3']-aa['recall@3']:+.3f}) at LLM cost {la['cost_llm']:.2f} vs {aa['cost_llm']:.2f}")

    out = {"n": len(rows), "overall": overall, "by_type": by_type, "tier_dist": tier_dist,
           "gate": th["_weakness_signals"], "answer_k": config.ANSWER_K}
    (config.ARTIFACTS_DIR / "policy_comparison_val.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> artifacts/policy_comparison_val.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
