"""v2.5 answer-quality eval with a SEMANTIC JUDGE (gpt-5.5, cross-provider).

EM penalizes correct-but-differently-phrased answers and is prior-inflated; a semantic judge
scores meaning, which is the fairer answer-quality read here. We regenerate each policy's answer
from its routed top-k (gentle stop) and judge correctness with gpt-5.5 on the test answerables,
for the hybrid baseline, the ladder, and always-decompose.

Phase 1 (single-thread) computes the routed top-k per policy (Qdrant + ONNX are not thread-safe).
Phase 2 (threaded) does the LLM answer-gen + gpt-5.5 judging (HTTP calls, thread-safe).

Usage (sandbox disabled):  python scripts/run_judge_eval.py
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

POLICIES = ["always_answer", "always_decompose", "ladder"]
PRETTY = {"always_answer": "hybrid baseline", "always_decompose": "always decompose", "ladder": "ladder"}


def main() -> int:
    corpus = data.load_corpus()
    th = sg.load_thresholds(path=config.ARTIFACTS_DIR / "thresholds_mixed.json")
    detector = th.get("_detector", sg.DEFAULT_DETECTOR)
    ircot = json.loads((config.ARTIFACTS_DIR / "ircot_mixed_test.json").read_text())
    test = [q for q in data.load_mixed_eval("test") if q.get("answerable") and q.get("gold_doc_ids")]

    # --- phase 1: routed top-k per policy (single-thread) ---
    print(f"phase 1: retrievals for {len(test)} answerable test queries ...")
    items = []
    for i, q in enumerate(test):
        enc = retrieval.encode_query(q["question"])
        base = retrieval.search(q["question"], mode="hybrid", k=config.TOP_K, fusion="rrf", encoded=enc)
        report = sg.read_signals(base, th, detector=detector)
        tier = 1 if report.healthy else (3 if policy_mod.looks_multi_hop(q["question"]) else 2)
        col = retrieval.colbert_search(q["question"], n_prefetch=config.RETRIEVE_N, k=config.TOP_K).doc_ids
        ir = ircot.get(q["id"], {}).get("doc_ids", base.doc_ids)
        ids = {"always_answer": base.doc_ids, "always_decompose": ir,
               "ladder": {1: base.doc_ids, 2: col, 3: ir}[tier]}
        items.append({"qid": q["id"], "qtype": q["query_type"], "question": q["question"],
                      "golds": ev.gold_answers(q), "tier": tier, "ids": ids})
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(test)} ...")

    def cands(doc_ids, k=config.ANSWER_K):
        return [retrieval.Candidate(doc_id=d, title=corpus.get(d, {}).get("title", ""),
                                    text=corpus.get(d, {}).get("text", ""), score=0.0, supports=[]) for d in doc_ids[:k]]

    # --- phase 2: gen + judge (threaded) ---
    tasks = [(it, p) for it in items for p in POLICIES]
    print(f"\nphase 2: gen + gpt-5.5 judge, {len(tasks)} (query x policy) tasks ...")
    results = {it["qid"]: {} for it in items}

    def work(it, p):
        ans, _ = agent.generate_answer(it["question"], cands(it["ids"][p]))
        stopped = agent.is_abstention(ans)
        j = 0 if stopped else ev.judge_answer(it["question"], ans, it["golds"])
        em = 0.0 if stopped else ev.best_answer_score(ans, it["golds"], ev.exact_match)
        return {"stopped": stopped, "judge": int(j), "em": em}

    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, it, p): (it["qid"], p) for it, p in tasks}
        for fut in as_completed(futs):
            qid, p = futs[fut]
            results[qid][p] = fut.result()
            done += 1
            if done % 60 == 0:
                print(f"  {done}/{len(tasks)} ...")

    rows = [{"qid": it["qid"], "qtype": it["qtype"], **{p: results[it["qid"]][p] for p in POLICIES}} for it in items]

    def agg(sub, p, metric):
        return round(st.mean(r[p][metric] for r in sub), 4)
    by = {}
    for p in POLICIES:
        by[p] = {"overall": {"judge": agg(rows, p, "judge"), "em": agg(rows, p, "em")},
                 **{qt: {"judge": agg([r for r in rows if r["qtype"] == qt], p, "judge"),
                        "em": agg([r for r in rows if r["qtype"] == qt], p, "em")}
                    for qt in ("single_hop", "multi_hop")}}

    # paired bootstrap CI on judge lift: ladder - baseline, ladder - decompose
    def boot(ref, seed=5252, n=2000):
        d = [r["ladder"]["judge"] - r[ref]["judge"] for r in rows]
        k = len(d); rng = random.Random(seed)
        bs = sorted(st.mean(d[rng.randrange(k)] for _ in range(k)) for _ in range(n))
        return {"lift": round(st.mean(d), 4), "ci95": [round(bs[int(.025 * n)], 4), round(bs[int(.975 * n)], 4)]}
    ci = {"vs_baseline": boot("always_answer"), "vs_decompose": boot("always_decompose")}

    out = {"n_answerable": len(rows), "judge_model": config.JUDGE_MODEL, "by_policy": by, "ci_ladder": ci}
    (config.ARTIFACTS_DIR / "judge_eval_v25.json").write_text(json.dumps(out, indent=2))

    print("\n=== answer quality: SEMANTIC JUDGE (gpt-5.5) vs EM, test answerable ===")
    print(f"{'policy':18s}{'judge':>8}{'EM':>8}{'judge single':>14}{'judge multi':>13}")
    for p in POLICIES:
        b = by[p]
        print(f"{PRETTY[p]:18s}{b['overall']['judge']:>8.3f}{b['overall']['em']:>8.3f}{b['single_hop']['judge']:>14.3f}{b['multi_hop']['judge']:>13.3f}")
    print(f"\nladder judge lift vs baseline:  {ci['vs_baseline']['lift']:+.4f}  CI {ci['vs_baseline']['ci95']}")
    print(f"ladder judge lift vs decompose: {ci['vs_decompose']['lift']:+.4f}  CI {ci['vs_decompose']['ci95']}")
    print("wrote -> artifacts/judge_eval_v25.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
