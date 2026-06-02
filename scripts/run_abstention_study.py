"""v2.5 abstention study (Decision B): gentle/system stop vs the haiku sufficiency
autorater, applied to the SAME ladder retrievals on the test slice.

The headline frontier (Decision A: routing) showed the ladder wins cost/quality on
ANSWERABLE retrieval. But escalating an unanswerable feeds the generator more context,
so the cheap gentle stop abstains LESS -> more false answers. This quantifies what a real
sufficiency check (the autorater) recovers, and what it costs in answer quality. The
lesson: self-evaluation for ROUTING is not the same as self-evaluation for STOPPING; pick
the stop that fits your abstention/answer tradeoff.

Reuses the cached IRCoT (ircot_mixed_test.json); routing is recomputed (LLM-free). Per
query we generate the answer once and run the autorater once, then score BOTH stop modes.

Usage:  python scripts/run_abstention_study.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import agent  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import eval as ev  # noqa: E402
import policy as policy_mod  # noqa: E402
import retrieval  # noqa: E402
import signals as sg  # noqa: E402

IRCOT_CACHE = config.ARTIFACTS_DIR / "ircot_mixed_test.json"


def main() -> int:
    corpus = data.load_corpus()
    th = sg.load_thresholds(path=config.ARTIFACTS_DIR / "thresholds_mixed.json")
    detector = th.get("_detector", sg.DEFAULT_DETECTOR)
    test = data.load_mixed_eval("test")
    ircot_cache = json.loads(IRCOT_CACHE.read_text()) if IRCOT_CACHE.exists() else {}
    assert ircot_cache, "run_headline_test must run first (populates ircot_mixed_test.json)"

    def cands(doc_ids, k=config.ANSWER_K):
        return [retrieval.Candidate(doc_id=d, title=corpus.get(d, {}).get("title", ""),
                                    text=corpus.get(d, {}).get("text", ""), score=0.0, supports=[])
                for d in doc_ids[:k]]

    rows = []
    for i, q in enumerate(test):
        question = q["question"]
        enc = retrieval.encode_query(question)
        base = retrieval.search(question, mode="hybrid", k=config.TOP_K, fusion="rrf", encoded=enc)
        report = sg.read_signals(base, th, detector=detector)
        tier = 1 if report.healthy else (3 if policy_mod.looks_multi_hop(question) else 2)
        if tier == 2:
            doc_ids = retrieval.colbert_search(question, n_prefetch=config.RETRIEVE_N, k=config.TOP_K).doc_ids
        elif tier == 3:
            doc_ids = ircot_cache.get(q["id"], {}).get("doc_ids", base.doc_ids)
        else:
            doc_ids = base.doc_ids
        c = cands(doc_ids)
        answer, _ = agent.generate_answer(question, c)
        gen_abstained = agent.is_abstention(answer)
        sufficient, _ = agent.sufficiency_judge(question, c)
        rows.append({"answerable": q.get("answerable"), "qtype": q["query_type"], "answer": answer,
                     "gen_abstained": gen_abstained, "autorater_sufficient": bool(sufficient),
                     "golds": ev.gold_answers(q), "question": question})
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(test)} ...")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    def score(stop_mode):
        # stopped? gentle = generator self-abstained; autorater = insufficient OR generator abstained
        def stopped(r):
            return r["gen_abstained"] if stop_mode == "gentle" else (not r["autorater_sufficient"] or r["gen_abstained"])
        ems = [0.0 if stopped(r) else ev.best_answer_score(r["answer"], r["golds"], ev.exact_match) for r in ans]
        f1s = [0.0 if stopped(r) else ev.best_answer_score(r["answer"], r["golds"], ev.token_f1) for r in ans]
        abstain_unans = st.mean(stopped(r) for r in unans)
        false_stop = st.mean(stopped(r) for r in ans)
        # abstention precision/recall/F1 vs MuSiQue answerable labels
        stops = [r for r in rows if stopped(r)]
        prec = st.mean(not r["answerable"] for r in stops) if stops else float("nan")
        rec = abstain_unans
        f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and (prec + rec) > 0) else float("nan")
        return {"em": round(st.mean(ems), 4), "f1": round(st.mean(f1s), 4),
                "abstain_rate_unans": round(abstain_unans, 4), "false_answer_unans": round(1 - abstain_unans, 4),
                "false_stop_answerable": round(false_stop, 4),
                "abstention_precision": round(prec, 4), "abstention_recall": round(rec, 4),
                "abstention_f1": round(f1, 4) if f1 == f1 else None,
                "em_single": round(st.mean(0.0 if stopped(r) else ev.best_answer_score(r["answer"], r["golds"], ev.exact_match)
                                           for r in ans if r["qtype"] == "single_hop"), 4),
                "em_multi": round(st.mean(0.0 if stopped(r) else ev.best_answer_score(r["answer"], r["golds"], ev.exact_match)
                                          for r in ans if r["qtype"] == "multi_hop"), 4)}

    out = {"n_answerable": len(ans), "n_unanswerable": len(unans),
           "gentle": score("gentle"), "autorater": score("autorater")}
    (config.ARTIFACTS_DIR / "abstention_study_v25.json").write_text(json.dumps(out, indent=2))

    print("\n=== Decision B: stop mechanism Pareto (test, ladder retrievals) ===")
    print(f"{'mode':10s} {'EM':>6s} {'F1':>6s} {'abst_F1':>8s} {'abstain':>8s} {'false_ans':>10s} {'false_stop':>11s}")
    for m in ("gentle", "autorater"):
        s = out[m]
        print(f"{m:10s} {s['em']:6.3f} {s['f1']:6.3f} {str(s['abstention_f1']):>8s} "
              f"{s['abstain_rate_unans']:8.3f} {s['false_answer_unans']:10.3f} {s['false_stop_answerable']:11.3f}")
    print(f"\ntradeoff: autorater recovers abstention "
          f"{out['gentle']['abstain_rate_unans']:.3f}->{out['autorater']['abstain_rate_unans']:.3f} "
          f"but EM {out['gentle']['em']:.3f}->{out['autorater']['em']:.3f}, "
          f"false_stop {out['gentle']['false_stop_answerable']:.3f}->{out['autorater']['false_stop_answerable']:.3f}")
    print("wrote -> artifacts/abstention_study_v25.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
