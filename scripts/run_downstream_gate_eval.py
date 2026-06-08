"""Downstream eval: does a CLASSIFIER gate route better than the OR-gate?

The detection study showed a classifier fusing signals barely beats the single signal on
GLOBAL AUC, but lifts MULTI-HOP weak-detection by ~+0.11 AUC. AUC is not the product metric.
This asks the decisive question: when each detector drives the ladder's tier-1 gate, does the
better multi-hop detection actually move the end-to-end COST/QUALITY frontier?

Method (LLM-free, like run_policy_comparison): counterfactual routing on the FROZEN TEST split.
For each query we have the retrieval under each action (baseline / ColBERT / cached IRCoT). Only
the TIER-1 GATE changes between conditions; the tier-2-vs-3 routing (looks_multi_hop) is identical,
so any difference is attributable to the gate. We score the ROUTED retrieval's precision
(full_gold@3, recall@3, MRR) against cost (mean retrieval + IRCoT sub-query calls).

Gates compared:
  always_answer / always_colbert / always_decompose   - fixed-policy frontier anchors
  or_gate                                              - the live v2.5 gate (dense_variance OR score_variance)
  lr2  / lr19 / lr19+looksmulti                        - classifier gates (threshold swept -> frontier)

Discipline: classifiers TRAINED on calibration features; thresholds for the matched-cost
head-to-head set on CALIBRATION; everything reported on TEST (touched once).

Usage:  python scripts/run_downstream_gate_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import config  # noqa: E402
import data  # noqa: E402
import policy as policy_mod  # noqa: E402
import retrieval  # noqa: E402
import signals as sg  # noqa: E402
import run_classifier_study as rcs  # noqa: E402

FEATURES = rcs.FEATURES
K, AK = config.TOP_K, config.ANSWER_K


def precision_scores(doc_ids, gold) -> dict:
    gs = set(gold or [])
    top1, top3 = set(doc_ids[:1]), set(doc_ids[:AK])
    rank = next((i + 1 for i, d in enumerate(doc_ids) if d in gs), 0)
    return {"recall@1": len(top1 & gs) / len(gs) if gs else float("nan"),
            "recall@3": len(top3 & gs) / len(gs) if gs else float("nan"),
            "full_gold@3": 1.0 if gs and gs.issubset(top3) else 0.0,
            "mrr": (1.0 / rank) if rank else 0.0}


def lr_pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=5000, class_weight="balanced"))])


def cost_of(tier, n_sub):
    # LLM cost is driven entirely by tier-3 (IRCoT) sub-query calls; tier 1/2 cost one retrieval.
    if tier == 3:
        return {"retr": 1 + n_sub, "llm": n_sub}
    return {"retr": 1, "llm": 0}


def main():
    import numpy as np
    th = sg.load_thresholds(path=ART / "thresholds_mixed.json")
    detector = th.get("_detector", sg.DEFAULT_DETECTOR)

    # ---- training data: calibration features (+ looks_multi_hop, the gold-free type proxy) ----
    cal = json.loads((ART / "features_ext_calibration.json").read_text())
    qtext = {r["id"]: r["question"] for r in data.load_questions_mixed()}
    for r in cal:
        r["_looksmulti"] = 1.0 if policy_mod.looks_multi_hop(qtext.get(r["question_id"], "")) else 0.0
    y_cal = np.array([1 - r["full_gold_label"] for r in cal])

    def Xc(feats):
        return np.array([[r[f] for f in feats] for r in cal], dtype=float)

    GATES = {"lr2": ["dense_variance", "score_variance"],
             "lr19": FEATURES,
             "lr19+looksmulti": FEATURES + ["_looksmulti"]}
    fitted, p_cal = {}, {}
    for name, feats in GATES.items():
        p = lr_pipe(); p.fit(Xc(feats), y_cal)
        fitted[name] = (p, feats)
        p_cal[name] = p.predict_proba(Xc(feats))[:, 1]
    # OR-gate calibration escalation rate (to match cost in the head-to-head)
    or_cal_escal = float(np.mean([1 - r["full_gold_label"] for r in cal]))  # placeholder; recomputed below from rule
    or_cal_fire = []
    sel = th.get("_weakness_signals") or ["dense_variance", "score_variance"]
    for r in cal:
        or_cal_fire.append(1 if any(r.get(s, float("inf")) < th[s] for s in sel) else 0)
    or_cal_escal = float(np.mean(or_cal_fire))

    # threshold per classifier matched to OR-gate's calibration escalation rate
    matched_thr = {}
    for name in GATES:
        # choose T so that mean(P_cal >= T) ~= or_cal_escal  (the (1-q)-quantile of P_cal)
        matched_thr[name] = float(np.quantile(p_cal[name], 1 - or_cal_escal))

    # ---- per-query test material (LLM-free: cached IRCoT) ----
    ircot = json.loads((ART / "ircot_mixed_test.json").read_text())
    test = [q for q in data.load_mixed_eval("test") if q.get("answerable") and q.get("gold_doc_ids")]
    print(f"frozen test answerable: {len(test)} "
          f"(single={sum(q['query_type']=='single_hop' for q in test)}, "
          f"multi={sum(q['query_type']=='multi_hop' for q in test)})")
    print(f"OR-gate cal escalation rate: {or_cal_escal:.3f}; matched classifier thresholds: "
          + ", ".join(f"{k}={v:.3f}" for k, v in matched_thr.items()))

    rows = []
    for i, q in enumerate(test):
        qid, gold, question = q["id"], q["gold_doc_ids"], q["question"]
        enc = retrieval.encode_query(question)
        base = retrieval.search(question, mode="hybrid", k=K, fusion="rrf", encoded=enc)
        col = retrieval.colbert_search(question, n_prefetch=config.RETRIEVE_N, k=K)
        ir = ircot[qid]
        feats_row = rcs.features_from_result(base)
        feats_row["_looksmulti"] = 1.0 if policy_mod.looks_multi_hop(question) else 0.0
        prec = {1: precision_scores(base.doc_ids, gold),
                2: precision_scores(col.doc_ids, gold),
                3: precision_scores(ir["doc_ids"], gold)}
        pweak = {name: float(p.predict_proba(np.array([[feats_row[f] for f in feats]]))[0, 1])
                 for name, (p, feats) in fitted.items()}
        or_healthy = sg.read_signals(base, th, detector=detector).healthy
        rows.append({"qtype": q["query_type"], "looksmulti": feats_row["_looksmulti"] == 1.0,
                     "n_sub": ir["n_sub"], "prec": prec, "pweak": pweak, "or_healthy": or_healthy})
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(test)} ...")

    # ---- aggregation helpers ----
    def route_tier(healthy, looksmulti):
        if healthy:
            return 1
        return 3 if looksmulti else 2

    def agg(decide_fn, subset=None):
        rs = rows if subset is None else [r for r in rows if r["qtype"] == subset]
        fg = rec = mrr = cr = cl = esc = dec = 0.0
        for r in rs:
            tier = decide_fn(r)
            p = r["prec"][tier]; c = cost_of(tier, r["n_sub"])
            fg += p["full_gold@3"]; rec += p["recall@3"]; mrr += p["mrr"]
            cr += c["retr"]; cl += c["llm"]; esc += (tier != 1); dec += (tier == 3)
        n = len(rs)
        return {"full_gold@3": round(fg / n, 4), "recall@3": round(rec / n, 4), "mrr": round(mrr / n, 4),
                "cost_retr": round(cr / n, 3), "cost_llm": round(cl / n, 3),
                "escalate_rate": round(esc / n, 3), "decompose_rate": round(dec / n, 3), "n": n}

    def or_decide(r):
        return route_tier(r["or_healthy"], r["looksmulti"])

    def lr_decide(name, thr):
        return lambda r: route_tier(r["pweak"][name] < thr, r["looksmulti"])

    fixed = {
        "always_answer": lambda r: 1,
        "always_colbert": lambda r: 2,
        "always_decompose": lambda r: 3,
    }

    out = {"n": len(rows), "split": "test (frozen)", "or_cal_escalation": round(or_cal_escal, 3),
           "matched_thresholds": {k: round(v, 4) for k, v in matched_thr.items()}}

    # 1) headline points (overall + by type)
    points = {}
    for name, fn in fixed.items():
        points[name] = {"overall": agg(fn), **{qt: agg(fn, qt) for qt in ("single_hop", "multi_hop")}}
    points["or_gate"] = {"overall": agg(or_decide), **{qt: agg(or_decide, qt) for qt in ("single_hop", "multi_hop")}}
    for name in GATES:
        fn = lr_decide(name, matched_thr[name])
        points[f"{name}@matched"] = {"overall": agg(fn), **{qt: agg(fn, qt) for qt in ("single_hop", "multi_hop")}}
    out["points"] = points

    # 2) classifier frontier (sweep threshold) - overall full_gold@3 vs cost
    frontiers = {}
    grid = [round(t, 3) for t in np.linspace(0.02, 0.98, 25)]
    for name in GATES:
        curve = []
        for t in grid:
            a = agg(lr_decide(name, t))
            curve.append({"thr": t, "cost_llm": a["cost_llm"], "cost_retr": a["cost_retr"],
                          "full_gold@3": a["full_gold@3"], "recall@3": a["recall@3"],
                          "decompose_rate": a["decompose_rate"]})
        frontiers[name] = curve
    out["frontiers"] = frontiers

    (ART / "downstream_gate_eval.json").write_text(json.dumps(out, indent=2))
    print("\nwrote artifacts/downstream_gate_eval.json")

    # ---- console summary ----
    def line(label, p):
        o = p["overall"]; m = p["multi_hop"]
        print(f"  {label:22s} fg@3={o['full_gold@3']:.3f} rec@3={o['recall@3']:.3f} | "
              f"multi fg@3={m['full_gold@3']:.3f} | cost_llm={o['cost_llm']:.2f} decomp={o['decompose_rate']:.2f}")
    print("\n=== gate cost/quality on TEST (overall + multi-hop) ===")
    for name in ["always_answer", "always_colbert", "always_decompose", "or_gate",
                 "lr2@matched", "lr19@matched", "lr19+looksmulti@matched"]:
        line(name, points[name])
    print("\n(matched = classifier threshold set on cal to match OR-gate escalation rate -> ~equal cost)")
    build_html(out)
    print("wrote downstream_gate_eval.html")
    return 0


# ---------------------------------------------------------------------------
def build_html(d):
    import html as _h

    def f3(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "&mdash;"

    P = d["points"]
    orr = P["or_gate"]

    def drow(label, key, hi=False):
        p = P[key]; o, s, m = p["overall"], p["single_hop"], p["multi_hop"]
        dfg = o["full_gold@3"] - orr["overall"]["full_gold@3"]
        dcls = "pos" if dfg > 0.0005 else ("neg" if dfg < -0.0005 else "flat")
        style = ' style="background:#fff6fa"' if hi else ""
        return (f"<tr{style}><td style='text-align:left'>{_h.escape(label)}</td>"
                f"<td>{f3(o['full_gold@3'])} <span class='{dcls}'>({'+' if dfg>=0 else ''}{dfg:.3f})</span></td>"
                f"<td>{f3(o['recall@3'])}</td><td>{f3(s['full_gold@3'])}</td><td>{f3(m['full_gold@3'])}</td>"
                f"<td>{o['cost_llm']:.2f}</td><td>{o['decompose_rate']:.2f}</td></tr>")

    body = ""
    for lbl, key in [("always_answer (tier 1)", "always_answer"), ("always_colbert (tier 2)", "always_colbert"),
                     ("always_decompose (tier 3)", "always_decompose"), ("OR-gate (live v2.5)", "or_gate"),
                     ("LR(2) @matched cost", "lr2@matched"), ("LR(19) @matched cost", "lr19@matched"),
                     ("LR(19)+looks_multi @matched", "lr19+looksmulti@matched")]:
        body += drow(lbl, key, hi=(key == "lr19@matched"))

    # frontier table: classifier full_gold@3 at a few decompose-cost levels vs OR-gate point
    fr = d["frontiers"]["lr19"]
    fr_rows = ""
    for pt in fr[::3]:
        fr_rows += (f"<tr><td>{pt['thr']:.2f}</td><td>{pt['cost_llm']:.2f}</td>"
                    f"<td>{pt['decompose_rate']:.2f}</td><td>{f3(pt['full_gold@3'])}</td><td>{f3(pt['recall@3'])}</td></tr>")
    orll = orr["overall"]

    css = """:root{--ink:#1a1a22;--muted:#5b6270;--line:#e5e7eb;--bg:#fafafb;--accent:#d6336c;--pos:#138a52;--neg:#c92a2a;}
*{box-sizing:border-box;}body{margin:0;font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);}
.wrap{max-width:1000px;margin:0 auto;padding:40px 48px 100px;}
h1{font-size:28px;margin:0 0 6px;}h2{font-size:21px;margin:40px 0 12px;padding-top:10px;border-top:1px solid var(--line);}
p,li{color:#23262e;}.sub{color:var(--muted);font-size:14px;}
.tag{display:inline-block;background:#fdeef4;color:var(--accent);border:1px solid #f6c6da;border-radius:999px;padding:2px 11px;font-size:12.5px;font-weight:600;margin-right:6px;}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13.5px;}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);}td:not(:first-child),th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums;}
.pos{color:var(--pos);font-weight:600;}.neg{color:var(--neg);font-weight:600;}.flat{color:var(--muted);}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:#f2f3f5;padding:1px 5px;border-radius:4px;}
.foot{color:var(--muted);font-size:13px;margin-top:50px;border-top:1px solid var(--line);padding-top:16px;}"""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Downstream gate eval</title>
<style>{css}</style></head><body><div class="wrap">
<div><span class="tag">Branch study</span><span class="tag">classifier-signal-fusion</span><span class="tag">downstream / end-to-end</span><span class="tag">test n={d['n']}</span></div>
<h1>Does a classifier gate route the ladder better than the OR-gate?</h1>
<p class="sub">Counterfactual routing on the frozen TEST split (LLM-free; cached IRCoT). Only the tier-1
gate changes; tier-2-vs-3 routing is identical, so differences are the gate's. Classifiers trained on
calibration; thresholds matched to the OR-gate's escalation rate on calibration (so cost is ~equal).
Quality = full_gold@3 of the ROUTED retrieval; cost = mean IRCoT sub-query (LLM) calls.</p>

<h2>Cost / quality at matched cost</h2>
<p>Deltas in parentheses are full_gold@3 vs the OR-gate. The classifier earns its place only if it
lifts quality at equal-or-lower cost - especially on multi-hop, where the detector AUC gain was.</p>
<table>
<tr><th>gate (test)</th><th>full_gold@3 (Δ vs OR)</th><th>recall@3</th><th>single fg@3</th><th>multi fg@3</th><th>cost (LLM)</th><th>decompose rate</th></tr>
{body}
</table>

<h2>LR(19) frontier: quality vs decompose cost (threshold swept)</h2>
<p>The OR-gate sits at cost_llm <b>{orll['cost_llm']:.2f}</b>, full_gold@3 <b>{f3(orll['full_gold@3'])}</b>,
decompose rate <b>{orll['decompose_rate']:.2f}</b>. Does the classifier curve clear that point?</p>
<table>
<tr><th>threshold</th><th>cost (LLM)</th><th>decompose rate</th><th>full_gold@3</th><th>recall@3</th></tr>
{fr_rows}
</table>

<h2>How to read this</h2>
<ul>
<li><b>This is the test detection AUC could not answer.</b> A gate is only as good as the routing
decisions it drives and the corrective action behind them. Equal-or-better full_gold@3 at equal-or-lower
cost = the classifier earns its place; otherwise the OR-gate (no model) wins on simplicity.</li>
<li><b>Watch the multi-hop column.</b> That is where the classifier's detection edge lived (+0.11 AUC).
If it does not convert to multi-hop full_gold@3 here, the AUC gain did not survive into routing value.</li>
<li><b>looks_multi version</b> feeds the gold-free hop-type proxy the router already uses - the
inference-available form of the type-awareness lever.</li>
</ul>
<div class="foot">Generated by <span class="mono">scripts/run_downstream_gate_eval.py</span> (LLM-free; cached IRCoT).
Numbers in <span class="mono">artifacts/downstream_gate_eval.json</span>.</div>
</div></body></html>"""
    (REPO / "downstream_gate_eval.html").write_text(doc)


if __name__ == "__main__":
    sys.exit(main())
