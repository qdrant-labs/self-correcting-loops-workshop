"""v2.5 CORRECTED headline (post-Codex adversarial review). Supersedes headline_test_v25.json.

Fixes the issues the review found, in one consistent pass:
  - HONEST metrics: lead with recall@3 and full_gold@3 (complete-evidence); MRR is reported but
    LABELED as first-gold (lenient on multi-hop), not the headline.
  - ALL fixed policies in the frontier: always_answer / always_colbert / always_rerank /
    always_decompose vs the ladder (rerank was missing before).
  - CORRECTED LLM cost: an IRCoT decompose spends min(n_sub+1, 3) sub-query LLM calls (the loop
    asks for the next sub-query even on the terminal ENOUGH hop), not n_sub. Retrieval (Qdrant)
    calls reported separately and matched to the loop (baseline runs before any tier action).
  - SELECTIVE ACCURACY over the FULL workload (incl. unanswerables): (answer-correct + correct
    abstention)/N, the combined-utility metric, for ladder vs always_answer.
  - CIs for ladder vs always_answer AND ladder vs always_decompose (the strongest fixed policy),
    on recall@3, full_gold@3, and EM.
  - One consistent answer-generation pass (no cross-artifact regeneration variance).
  - touched_once = FALSE, with the measured reuse counts vs the v1/v2 test manifests.

Saves per-query data so CIs / metrics can be recomputed without re-running.
Usage:  python scripts/run_headline_final.py   (IRCoT reused from ircot_mixed_test.json)
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import agent  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import eval as ev  # noqa: E402
import policy as policy_mod  # noqa: E402
import retrieval  # noqa: E402
import signals as sg  # noqa: E402

IRCOT_CACHE = config.ARTIFACTS_DIR / "ircot_mixed_test.json"
ANSWER_POLICIES = ["always_answer", "always_decompose", "ladder"]   # get generated answers
FRONTIER_POLICIES = ["always_answer", "always_colbert", "always_rerank", "always_decompose", "ladder"]
PRETTY = {"always_answer": "always answer", "always_colbert": "always ColBERT",
          "always_rerank": "always rerank", "always_decompose": "always decompose", "ladder": "ladder"}


def ircot_llm_calls(n_sub):
    """The IRCoT loop calls _next_subquery min(n_sub+1, 3) times (it asks once more, gets ENOUGH,
    unless it hit the hop cap). config: range(1,4) -> 3 max hops."""
    return min(n_sub + 1, 3)


def precision_scores(doc_ids, gold):
    gs = set(gold or [])
    rank = next((i + 1 for i, d in enumerate(doc_ids) if d in gs), 0)
    return {"recall@1": len(set(doc_ids[:1]) & gs) / len(gs) if gs else float("nan"),
            "recall@3": len(set(doc_ids[:3]) & gs) / len(gs) if gs else float("nan"),
            "full_gold@3": 1.0 if gs and gs.issubset(set(doc_ids[:3])) else 0.0,
            "mrr_first": (1.0 / rank) if rank else 0.0}


def main() -> int:
    corpus = data.load_corpus()
    th = sg.load_thresholds(path=config.ARTIFACTS_DIR / "thresholds_mixed.json")
    detector = th.get("_detector", sg.DEFAULT_DETECTOR)
    test = data.load_mixed_eval("test")
    by_id = {q["id"]: q for q in test}
    ircot_cache = json.loads(IRCOT_CACHE.read_text()) if IRCOT_CACHE.exists() else {}

    # reuse disclosure vs prior test manifests
    man = json.loads((config.ARTIFACTS_DIR / "mixed_manifest.json").read_text())["splits"]["test"]
    reuse = {}
    for tag, fn in (("v2", "test_manifest_v2.json"), ("v1", "test_manifest.json")):
        p = config.ARTIFACTS_DIR / fn
        if p.exists():
            m = json.loads(p.read_text())
            pa, pu = set(m.get("answerable_ids", [])), set(m.get("unanswerable_ids", []))
            reuse[tag] = {"multi_in_prior_ans": len(set(man["multi"]) & pa),
                          "unans_in_prior_unans": len(set(man["unanswerable"]) & pu),
                          "single_src_in_prior_ans": len({i.rsplit("__h", 1)[0] for i in man["single"]} & pa)}

    def cands(doc_ids, k=config.ANSWER_K):
        return [retrieval.Candidate(doc_id=d, title=corpus.get(d, {}).get("title", ""),
                                    text=corpus.get(d, {}).get("text", ""), score=0.0, supports=[]) for d in doc_ids[:k]]

    rows = []
    for i, q in enumerate(test):
        question, gold, ans = q["question"], q.get("gold_doc_ids", []), q.get("answerable")
        enc = retrieval.encode_query(question)
        base = retrieval.search(question, mode="hybrid", k=config.TOP_K, fusion="rrf", encoded=enc)
        report = sg.read_signals(base, th, detector=detector)
        tier = 1 if report.healthy else (3 if policy_mod.looks_multi_hop(question) else 2)
        col = retrieval.colbert_search(question, n_prefetch=config.RETRIEVE_N, k=config.TOP_K)
        rer = retrieval.rerank(base, query=question, k=config.TOP_K)
        if q["id"] not in ircot_cache:
            res, _ = agent.ircot_search(question, k=config.TOP_K, fusion="rrf", encoded=enc)
            ircot_cache[q["id"]] = {"doc_ids": res.doc_ids, "n_sub": len(res.sub_queries)}
        ir = ircot_cache[q["id"]]
        ids = {"always_answer": base.doc_ids, "always_colbert": col.doc_ids, "always_rerank": rer.doc_ids,
               "always_decompose": ir["doc_ids"], "ladder": {1: base.doc_ids, 2: col.doc_ids, 3: ir["doc_ids"]}[tier]}
        n_sub = ir["n_sub"]
        # corrected cost per policy: (qdrant_calls, llm_calls). baseline retrieval always runs first.
        cost = {"always_answer": (1, 0), "always_colbert": (2, 0), "always_rerank": (1, 0),
                "always_decompose": (2 + n_sub, ircot_llm_calls(n_sub)),
                "ladder": {1: (1, 0), 2: (2, 0), 3: (2 + n_sub, ircot_llm_calls(n_sub))}[tier]}
        row = {"qid": q["id"], "qtype": q["query_type"], "answerable": ans, "tier": tier, "ids": ids, "cost": cost}
        if ans and gold:
            row["prec"] = {p: precision_scores(ids[p], gold) for p in FRONTIER_POLICIES}
        rows.append(row)
        if (i + 1) % 30 == 0:
            print(f"  retrieval {i+1}/{len(test)} ...")
    IRCOT_CACHE.write_text(json.dumps(ircot_cache, indent=2))

    # one consistent answer pass for the 3 policies on ALL test queries
    print("\ngenerating answers (always_answer, always_decompose, ladder), one consistent pass ...")
    for j, r in enumerate(rows):
        q = by_id[r["qid"]]
        r["answer"], r["stopped"], r["em"] = {}, {}, {}
        for p in ANSWER_POLICIES:
            txt, _ = agent.generate_answer(q["question"], cands(r["ids"][p]))
            stopped = agent.is_abstention(txt)
            r["answer"][p] = txt; r["stopped"][p] = stopped
            r["em"][p] = 0.0 if stopped else ev.best_answer_score(txt, ev.gold_answers(q), ev.exact_match)
            r["f1_" + p] = 0.0 if stopped else ev.best_answer_score(txt, ev.gold_answers(q), ev.token_f1)
        if (j + 1) % 40 == 0:
            print(f"  answers {j+1}/{len(rows)} ...")

    arows = [r for r in rows if r["answerable"] and "prec" in r]
    urows = [r for r in rows if not r["answerable"]]
    METRICS = ["recall@1", "recall@3", "full_gold@3", "mrr_first"]

    def frontier(sub):
        out = {}
        for p in FRONTIER_POLICIES:
            d = {m: round(st.mean(r["prec"][p][m] for r in sub), 4) for m in METRICS}
            d["qdrant_calls"] = round(st.mean(r["cost"][p][0] for r in sub), 3)
            d["llm_calls"] = round(st.mean(r["cost"][p][1] for r in sub), 3)
            out[p] = d
        return out

    overall = frontier(arows)
    by_type = {qt: frontier([r for r in arows if r["qtype"] == qt]) for qt in ("single_hop", "multi_hop")}

    # answer EM/F1 by type for the 3 answer policies
    def ablock(sub, p):
        return {"n": len(sub), "em": round(st.mean(r["em"][p] for r in sub), 4),
                "f1": round(st.mean(r["f1_" + p] for r in sub), 4)}
    answers = {p: {"overall": ablock(arows, p),
                   **{qt: ablock([r for r in arows if r["qtype"] == qt], p) for qt in ("single_hop", "multi_hop")}}
               for p in ANSWER_POLICIES}

    # abstention + selective accuracy over the FULL workload (ladder + always_answer)
    sel = {}
    for p in ("always_answer", "ladder", "always_decompose") if all("always_decompose" in r["em"] for r in urows) else ("always_answer", "ladder"):
        ans_correct = sum(r["em"][p] >= 1.0 for r in arows)
        unans_abstain = sum(r["stopped"][p] for r in urows)
        sel[p] = {"selective_accuracy": round((ans_correct + unans_abstain) / len(rows), 4),
                  "abstain_rate_unans": round(st.mean(r["stopped"][p] for r in urows), 4),
                  "answerable_em": round(st.mean(r["em"][p] for r in arows), 4)}

    # CIs: ladder vs always_answer AND vs always_decompose, on recall@3 / full_gold@3 / em (answerable)
    def boot(metric_of, ref, seed=5252, n=2000):
        d = [metric_of(r, "ladder") - metric_of(r, ref) for r in arows]
        k = len(d); rng = random.Random(seed)
        bs = sorted(st.mean(d[rng.randrange(k)] for _ in range(k)) for _ in range(n))
        return {"lift": round(st.mean(d), 4), "ci95": [round(bs[int(.025 * n)], 4), round(bs[int(.975 * n)], 4)]}
    metric_funcs = {"recall@3": lambda r, p: r["prec"][p]["recall@3"],
                    "full_gold@3": lambda r, p: r["prec"][p]["full_gold@3"],
                    "em": lambda r, p: r["em"][p]}
    cis = {ref: {m: boot(fn, ref) for m, fn in metric_funcs.items()} for ref in ("always_answer", "always_decompose")}

    out = {"n_test": len(test), "n_answerable": len(arows), "n_unanswerable": len(urows),
           "touched_once": False,
           "test_reuse": reuse,
           "lead_metrics": ["recall@3", "full_gold@3"],
           "note_mrr": "mrr_first = reciprocal rank of the FIRST gold doc; lenient on multi-hop (rewards any one support passage). NOT the headline metric.",
           "overall": overall, "by_type": by_type, "answers": answers,
           "selective_accuracy": sel, "tier_dist": {qt: {t: sum(1 for r in rows if r["qtype"] == qt and r["tier"] == t) for t in (1, 2, 3)} for qt in ("single_hop", "multi_hop", "unanswerable")},
           "ci_vs": cis, "gate": th["_weakness_signals"], "answer_k": config.ANSWER_K}
    (config.ARTIFACTS_DIR / "headline_final_v25.json").write_text(json.dumps(out, indent=2))
    (config.ARTIFACTS_DIR / "headline_final_perquery.json").write_text(json.dumps(
        [{k: r[k] for k in ("qid", "qtype", "answerable", "tier", "prec", "em", "stopped", "cost") if k in r} for r in rows], indent=2))

    # report
    def ptable(title, a):
        print(f"\n=== {title} ===")
        print(f"{'policy':18s}{'rec@1':>7}{'rec@3':>7}{'fg@3':>7}{'MRR*':>7}{'qdrant':>8}{'llm':>6}")
        for p in FRONTIER_POLICIES:
            v = a[p]; print(f"{PRETTY[p]:18s}{v['recall@1']:7.3f}{v['recall@3']:7.3f}{v['full_gold@3']:7.3f}{v['mrr_first']:7.3f}{v['qdrant_calls']:8.2f}{v['llm_calls']:6.2f}")
    ptable("CORRECTED frontier - OVERALL (answerable)", overall)
    ptable("single-hop", by_type["single_hop"]); ptable("multi-hop", by_type["multi_hop"])
    print("\n=== answer EM (one consistent pass) ===")
    for p in ANSWER_POLICIES:
        print(f"  {PRETTY[p]:18s} EM {answers[p]['overall']['em']:.3f}  (single {answers[p]['single_hop']['em']:.3f}, multi {answers[p]['multi_hop']['em']:.3f})")
    print("\n=== selective accuracy over FULL workload (answer-correct + correct-abstention)/N ===")
    for p, v in sel.items():
        print(f"  {PRETTY[p]:18s} {v['selective_accuracy']:.3f}  (answerable EM {v['answerable_em']:.3f}, abstain unans {v['abstain_rate_unans']:.3f})")
    print("\n=== ladder lift CIs (paired bootstrap, answerable) ===")
    for ref in ("always_answer", "always_decompose"):
        print(f"  vs {ref}:")
        for m in ("recall@3", "full_gold@3", "em"):
            c = cis[ref][m]; sigflag = "clears 0" if c['ci95'][0] > 0 else ("clears 0 (neg)" if c['ci95'][1] < 0 else "crosses 0")
            print(f"     {m:11s} {c['lift']:+.4f}  CI [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]  {sigflag}")
    print(f"\ntest reuse (touched_once=False): {reuse}")
    print("wrote -> artifacts/headline_final_v25.json (+ _perquery.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
