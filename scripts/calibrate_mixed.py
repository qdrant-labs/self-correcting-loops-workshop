"""v2.5 Step 3: signal selection + threshold calibration on the MIXED workload, in the
PRECISION regime.

Decision A's tier-1 confidence gate answers when retrieval is strong enough to answer from
the FOCUSED top-ANSWER_K context. So the "good" LABEL is full_gold@ANSWER_K (all gold within
the top-ANSWER_K of the fused ranking) - a precision label, not recall@10. Under this label
the spread of the RAW DENSE cosine scores (dense_gap) is a much stronger weakness signal than
any RRF-fusion signal (RRF rank-compression flattens the v2 height/spread signals to ~0.51).

Eval integrity (unchanged): features per split; SELECT signals on VALIDATION; calibrate the
operating THRESHOLD on CALIBRATION; the test slice is never touched here. Thresholds are
corpus/retriever/regime-specific - NOT portable.

Writes:
  artifacts/features_mixed_{cal,val}.json  - the feature matrix (cached; reused on re-run)
  artifacts/signal_analysis_mixed.json     - AUC table (overall + by query type), correlations, selection
  artifacts/thresholds_mixed.json          - the FROZEN tier-1 gate (signal + threshold + metadata)

Usage:
  python scripts/calibrate_mixed.py          # full pass (recomputes features if absent)
  python scripts/calibrate_mixed.py --recompute
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import config  # noqa: E402
import data  # noqa: E402
import eval as ev  # noqa: E402
import retrieval  # noqa: E402
import signals as sg  # noqa: E402

# Candidate weakness signals (value -> oriented so HIGHER == more likely BAD via _badness).
# dense_* read raw dense cosines; the fusion signals read the RRF score; divergence is pre-fusion.
FIRE_LOW = ("dense_gap", "dense_variance", "confidence_gap", "score_variance", "max_score", "evidence_coverage")
DETECTORS = (config.BM25_VEC, config.MINICOIL_VEC)
LABEL_K = config.ANSWER_K  # full_gold@ANSWER_K = answerable from the focused context


def feature_row(q: dict) -> dict:
    res = retrieval.search(q["question"], mode="hybrid", k=config.TOP_K, fusion="rrf")
    vals = sg.signal_values(res, detector=config.BM25_VEC, k=config.TOP_K)
    gold = set(q.get("gold_doc_ids", []))
    fg_at_label = 1 if gold and gold.issubset(set(res.doc_ids[:LABEL_K])) else 0
    row = {
        "question_id": q["id"],
        "query_type": q["query_type"],
        "n_hops": q.get("n_hops"),
        "full_gold_label": fg_at_label,                 # full_gold@ANSWER_K (the precision label)
        "full_gold10": 1 if gold and gold.issubset(set(res.doc_ids[:10])) else 0,
    }
    for s in FIRE_LOW:
        row[s] = vals[s]
    for det in DETECTORS:
        row[f"divergence_{det}"] = sg.divergence(res.raw, det, config.TOP_K, "overlap")
    return row


def build_features(split: str) -> list[dict]:
    qs = [q for q in data.load_mixed_eval(split) if q.get("answerable") and q.get("gold_doc_ids")]
    return [feature_row(q) for q in qs]


def _auc(rows, signal, fires_above):
    from sklearn.metrics import roc_auc_score
    y = [1 - r["full_gold_label"] for r in rows]
    if len(set(y)) < 2:
        return float("nan")
    x = [r[signal] for r in rows]
    x = [v if fires_above else -v for v in x]
    pairs = [(a, b) for a, b in zip(y, x) if not (isinstance(b, float) and math.isnan(b))]
    if len({p[0] for p in pairs}) < 2:
        return float("nan")
    return round(float(roc_auc_score([p[0] for p in pairs], [p[1] for p in pairs])), 4)


def auc_table(rows: list[dict], detector: str) -> dict:
    out = {}
    for s in FIRE_LOW:
        out[s] = _auc(rows, s, fires_above=False)
    out["retriever_divergence"] = _auc(rows, f"divergence_{detector}", fires_above=True)
    return out


def by_type(rows, detector):
    res = {}
    for qt in ("single_hop", "multi_hop"):
        sub = [r for r in rows if r["query_type"] == qt]
        res[qt] = {"n": len(sub), "base_rate": round(sum(r["full_gold_label"] for r in sub) / len(sub), 3) if sub else None,
                   "auc": auc_table(sub, detector)}
    return res


def correlations(rows: list[dict], detector: str) -> dict:
    import numpy as np
    names = list(FIRE_LOW) + ["retriever_divergence"]
    cols = []
    for n in names:
        key = f"divergence_{detector}" if n == "retriever_divergence" else n
        cols.append([r[key] for r in rows])
    mat = np.corrcoef(np.array(cols))
    return {"labels": names, "matrix": [[round(float(v), 3) for v in row] for row in mat]}


def select(val_auc: dict, corr: dict, min_auc=0.62, corr_thresh=0.85) -> dict:
    labels = corr["labels"]; idx = {n: i for i, n in enumerate(labels)}; mat = corr["matrix"]
    kept = [s for s in labels if isinstance(val_auc.get(s), (int, float)) and val_auc[s] == val_auc[s] and val_auc[s] >= min_auc]
    final, dropped, reasons = [], [], {}
    for s in labels:
        if s not in kept:
            dropped.append(s); reasons[s] = f"dropped: val AUC {val_auc.get(s)} < {min_auc}"
    for s in sorted(kept, key=lambda n: -val_auc[n]):
        red = next((k for k in final if abs(mat[idx[s]][idx[k]]) > corr_thresh), None)
        if red:
            dropped.append(s); reasons[s] = f"dropped: |corr| {abs(mat[idx[s]][idx[red]]):.2f} with {red} (redundant; kept higher-AUC)"
        else:
            final.append(s); reasons[s] = f"kept: val AUC {val_auc[s]:.3f}"
    return {"weakness_signals": final, "dropped": dropped, "reasons": reasons}


def pick_threshold(rows, signal, fires_above):
    y = [1 - r["full_gold_label"] for r in rows]
    vals = [r[signal] for r in rows]
    pairs = [(yb, v) for yb, v in zip(y, vals) if not (isinstance(v, float) and math.isnan(v))]
    if len({p[0] for p in pairs}) < 2:
        return {"threshold": float("nan")}
    best = None
    for thr in sorted({v for _, v in pairs}):
        pred = [(v > thr) if fires_above else (v < thr) for _, v in pairs]
        tp = sum(1 for (yb, _), p in zip(pairs, pred) if p and yb == 1)
        fp = sum(1 for (yb, _), p in zip(pairs, pred) if p and yb == 0)
        fn = sum(1 for (yb, _), p in zip(pairs, pred) if not p and yb == 1)
        tn = sum(1 for (yb, _), p in zip(pairs, pred) if not p and yb == 0)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if best is None or j > best["youden_j"]:
            best = {"threshold": round(float(thr), 5), "youden_j": round(j, 4),
                    "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0, "recall": round(tpr, 4), "fpr": round(fpr, 4)}
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    feats = {}
    for split in ("calibration", "validation"):
        p = config.ARTIFACTS_DIR / f"features_mixed_{split[:3]}.json"
        if p.exists() and not args.recompute:
            feats[split] = json.loads(p.read_text())
            print(f"loaded cached features: {split} (n={len(feats[split])})")
        else:
            print(f"computing features: {split} ...")
            feats[split] = build_features(split)
            p.write_text(json.dumps(feats[split], indent=2))
            print(f"  wrote {p} (n={len(feats[split])})")

    cal, val = feats["calibration"], feats["validation"]

    # detector: bm25 vs minicoil by validation detect-bad AUC
    det_auc = {det: _auc(val, f"divergence_{det}", fires_above=True) for det in DETECTORS}
    detector = max(det_auc, key=lambda d: (det_auc[d] if det_auc[d] == det_auc[d] else 0))

    cal_auc = auc_table(cal, detector)
    val_auc = auc_table(val, detector)
    corr = correlations(val, detector)
    sel = select(val_auc, corr)

    analysis = {
        "label": f"full_gold@{LABEL_K}",
        "answer_k": config.ANSWER_K,
        "detector_auc": det_auc, "chosen_detector": detector,
        "auc_calibration": cal_auc,
        "auc_validation": val_auc,
        "auc_validation_by_type": by_type(val, detector),
        "correlations": corr,
        "selection": sel,
        "base_rate_validation": {
            "all": round(sum(1 - r["full_gold_label"] for r in val) / len(val), 3),
        },
    }
    (config.ARTIFACTS_DIR / "signal_analysis_mixed.json").write_text(json.dumps(analysis, indent=2))

    # calibrate operating thresholds on CALIBRATION for the selected signals + all candidates (for the trace)
    th = dict(sg.DEFAULT_THRESHOLDS)
    operating = {}
    for s in list(FIRE_LOW) + ["retriever_divergence"]:
        fires_above = (s == "retriever_divergence")
        key = f"divergence_{detector}" if fires_above else s
        op = pick_threshold(cal, key, fires_above)
        operating[s] = op
        if op.get("threshold") == op.get("threshold"):  # not NaN
            th[s] = op["threshold"]
    th["retriever_divergence"] = operating["retriever_divergence"].get("threshold", th["retriever_divergence"])
    th["_weakness_signals"] = sel["weakness_signals"] or ["dense_gap"]
    th["_detector"] = detector
    th["_fusion"] = "rrf"
    th["_label"] = f"full_gold@{LABEL_K}"
    th["_answer_k"] = config.ANSWER_K
    th["_calibrated"] = True
    th["_operating_points"] = operating
    (config.ARTIFACTS_DIR / "thresholds_mixed.json").write_text(json.dumps(th, indent=2))

    # report
    print(f"\n=== signal AUC (detect full_gold@{LABEL_K}==0); detector={detector} ===")
    print(f"{'signal':22s} {'cal':>7s} {'val':>7s} {'val/single':>11s} {'val/multi':>10s}")
    bt = analysis["auc_validation_by_type"]
    for s in sorted(val_auc, key=lambda n: -(val_auc[n] if val_auc[n] == val_auc[n] else 0)):
        print(f"{s:22s} {cal_auc.get(s,float('nan')):7.3f} {val_auc[s]:7.3f} "
              f"{bt['single_hop']['auc'].get(s,float('nan')):11.3f} {bt['multi_hop']['auc'].get(s,float('nan')):10.3f}")
    print(f"\nselected weakness signals: {sel['weakness_signals']}")
    for s, why in sel["reasons"].items():
        print(f"  {s}: {why}")
    print(f"\ntier-1 gate threshold(s):")
    for s in th["_weakness_signals"]:
        print(f"  {s} fires below {th[s]}  (cal Youden J={operating[s].get('youden_j')}, "
              f"prec={operating[s].get('precision')}, rec={operating[s].get('recall')})")
    print(f"\nwrote -> artifacts/{{signal_analysis_mixed,thresholds_mixed}}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
