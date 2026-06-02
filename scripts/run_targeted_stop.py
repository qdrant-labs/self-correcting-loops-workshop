"""v2.5 experiment: a TARGETED stop. Does applying the autorater only on ESCALATED queries
(tier 2/3) - keeping the cheap gentle stop on confident tier-1 answers - recover the abstention
the ladder loses on unanswerables, without over-abstaining on the easy answerables?

Reuses headline_final_perquery.json (per-query tier, ladder EM, gentle-stop verdict, and the
baseline's EM + gentle-stop). The only new work: the autorater (sufficiency) verdict on each
query's ROUTED top-k. The answer text is unchanged across stop variants, so EM under a variant =
0 if the variant stops, else the ladder's EM. We then recompute full-workload selective accuracy
(answer-correct on answerables + correct-abstention on unanswerables, over all queries) for:
  baseline (hybrid + gentle) | ladder + gentle | ladder + autorater-everywhere | ladder + TARGETED.

Usage (sandbox disabled):  python scripts/run_targeted_stop.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import agent  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import policy as policy_mod  # noqa: E402
import retrieval  # noqa: E402
import signals as sg  # noqa: E402


def main() -> int:
    corpus = data.load_corpus()
    th = sg.load_thresholds(path=config.ARTIFACTS_DIR / "thresholds_mixed.json")
    detector = th.get("_detector", sg.DEFAULT_DETECTOR)
    perq = {r["qid"]: r for r in json.loads((config.ARTIFACTS_DIR / "headline_final_perquery.json").read_text())}
    ircot = json.loads((config.ARTIFACTS_DIR / "ircot_mixed_test.json").read_text())
    test = data.load_mixed_eval("test")

    def cands(doc_ids, k=config.ANSWER_K):
        return [retrieval.Candidate(doc_id=d, title=corpus.get(d, {}).get("title", ""),
                                    text=corpus.get(d, {}).get("text", ""), score=0.0, supports=[]) for d in doc_ids[:k]]

    rows = []
    for i, q in enumerate(test):
        pq = perq[q["id"]]
        tier = pq["tier"]
        enc = retrieval.encode_query(q["question"])
        base = retrieval.search(q["question"], mode="hybrid", k=config.TOP_K, fusion="rrf", encoded=enc)
        if tier == 2:
            ids = retrieval.colbert_search(q["question"], n_prefetch=config.RETRIEVE_N, k=config.TOP_K).doc_ids
        elif tier == 3:
            ids = ircot.get(q["id"], {}).get("doc_ids", base.doc_ids)
        else:
            ids = base.doc_ids
        sufficient, _ = agent.sufficiency_judge(q["question"], cands(ids))
        rows.append({
            "answerable": q.get("answerable"), "qtype": q["query_type"], "tier": tier,
            "em_ladder": pq["em"]["ladder"], "gentle_ladder": pq["stopped"]["ladder"],
            "em_base": pq["em"]["always_answer"], "gentle_base": pq["stopped"]["always_answer"],
            "autorater_insufficient": (not sufficient),
        })
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(test)} ...")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    n = len(rows)

    def variant(stop_fn, em_of):
        # stop_fn(r) -> bool stopped; em_of(r) -> ladder/baseline EM when answered
        correct = sum((not stop_fn(r)) and em_of(r) >= 1.0 for r in ans) + sum(stop_fn(r) for r in unans)
        return {
            "selective_accuracy": round(correct / n, 4),
            "answerable_em": round(st.mean(0.0 if stop_fn(r) else em_of(r) for r in ans), 4),
            "abstain_unans": round(st.mean(stop_fn(r) for r in unans), 4),
            "false_stop_ans": round(st.mean(stop_fn(r) for r in ans), 4),
        }

    base_em = lambda r: r["em_base"]
    lad_em = lambda r: r["em_ladder"]
    variants = {
        "baseline_hybrid_gentle": variant(lambda r: r["gentle_base"], base_em),
        "ladder_gentle": variant(lambda r: r["gentle_ladder"], lad_em),
        "ladder_autorater_all": variant(lambda r: r["autorater_insufficient"] or r["gentle_ladder"], lad_em),
        "ladder_targeted": variant(
            lambda r: r["gentle_ladder"] if r["tier"] == 1 else (r["autorater_insufficient"] or r["gentle_ladder"]),
            lad_em),
    }
    out = {"n": n, "n_answerable": len(ans), "n_unanswerable": len(unans),
           "tier_dist": {t: sum(1 for r in rows if r["tier"] == t) for t in (1, 2, 3)}, "variants": variants}
    (config.ARTIFACTS_DIR / "targeted_stop_v25.json").write_text(json.dumps(out, indent=2))

    print("\n=== stop variants on the FULL test workload (n=%d) ===" % n)
    print(f"{'variant':28s}{'selective_acc':>14}{'ans_EM':>9}{'abstain_unans':>15}{'false_stop':>12}")
    for k, v in variants.items():
        print(f"{k:28s}{v['selective_accuracy']:>14.3f}{v['answerable_em']:>9.3f}{v['abstain_unans']:>15.3f}{v['false_stop_ans']:>12.3f}")
    b = variants["baseline_hybrid_gentle"]["selective_accuracy"]
    t = variants["ladder_targeted"]["selective_accuracy"]
    print(f"\nTARGETED ladder selective accuracy {t:.3f} vs baseline {b:.3f}  ->  "
          f"{'ladder WINS' if t > b else ('TIE' if abs(t-b) < 0.005 else 'baseline still wins')} "
          f"({t-b:+.3f})")
    print("wrote -> artifacts/targeted_stop_v25.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
