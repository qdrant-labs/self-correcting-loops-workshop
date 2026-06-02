"""Generate the demo traces the notebook renders (artifacts/demo_traces_v25.jsonl).

These four are illustrative, hand-picked from the VALIDATION split (so the held-out test
slice is never touched for a demo). Each shows one rung of the ladder, end to end, by
running the real self-correcting loop (agent.run_loop with the gentle/system stop and the
calibrated thresholds) and serializing the trace:

  2hop__101521_42157__h0   single-hop, confident  -> tier 1 ANSWER (the cheap path)
  2hop__130545_45439__h0   single-hop, weak       -> tier 2 ColBERT precision fix
  2hop__10253_65518        multi-hop, weak        -> tier 3 IRCoT decompose
  2hop__108098_170204      unanswerable           -> decompose, then STOP (gentle stop)

Re-running regenerates the file. The rung each query takes is deterministic (the gate
reads deterministic retrieval scores); the trace text (sub-questions, answer, latencies)
can vary slightly run to run because the agent/answer LLM calls are not bit-stable.

Usage:  python scripts/make_demo_traces.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import agent  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import signals as sg  # noqa: E402

DEMO_IDS = [
    "2hop__101521_42157__h0",   # tier 1: confident single-hop lookup
    "2hop__130545_45439__h0",   # tier 2: weak single-hop -> ColBERT
    "2hop__10253_65518",        # tier 3: weak multi-hop -> decompose
    "2hop__108098_170204",      # unanswerable -> decompose then STOP
]


def main() -> int:
    th = sg.load_thresholds(path=config.ARTIFACTS_DIR / "thresholds_mixed.json")
    by_id = {q["id"]: q for q in data.load_questions_mixed()}
    out = config.ARTIFACTS_DIR / "demo_traces_v25.jsonl"

    traces = []
    for qid in DEMO_IDS:
        q = by_id.get(qid)
        if q is None:
            raise SystemExit(f"demo id {qid!r} not found in questions_mixed.jsonl")
        tr = agent.run_loop(q["question"], qid=qid, answerable=q.get("answerable"),
                            gold_doc_ids=q.get("gold_doc_ids", []), thresholds=th, stop_mode="system")
        print(f"  {qid:30s} {q['query_type']:12s} -> {[s.action for s in tr.steps]}  stopped={tr.stopped}")
        traces.append(tr.to_dict())

    with out.open("w") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote -> {out} ({len(traces)} demo traces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
