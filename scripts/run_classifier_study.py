"""Classifier signal-fusion study  (branch: classifier-signal-fusion).

The CTO's question: instead of thresholding ONE hand-picked weakness signal, can a
CLASSIFIER that FUSES several signals detect weak retrieval better - and is the lift
worth the added complexity?

Same eval discipline as scripts/calibrate_mixed.py (do not break it):
  - label: full_gold@ANSWER_K == 0  (retrieval is WEAK = not all gold inside the focused top-K)
  - features per split; TRAIN on calibration, SELECT/compare on validation, REPORT once on test
  - thresholds/models are corpus-specific, never portable

Goes further than a single LR (Dylan's call, "take it far"):
  - engineers ~12 NEW cheap candidate signals from the SAME retrieval result (no extra queries)
  - ranks EVERY signal (old + new) by validation AUC to see if anything beats dense_variance
  - benchmarks single-signal thresholds and the live OR-gate against L2/L1 logistic regression
    and gradient boosting, on AUC, PR-AUC, by-type AUC, 5-fold CV stability, and a matched
    operating point (precision / recall / false-escalation rate)

Writes:
  artifacts/features_ext_{cal,val,test}.json   extended feature matrix (cached; reused on re-run)
  artifacts/classifier_study.json              every number this report shows
  classifier_study.html                        the review page

Usage:
  python scripts/run_classifier_study.py            # builds val/test features live if absent
  python scripts/run_classifier_study.py --recompute
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import config  # noqa: E402
import data  # noqa: E402
import retrieval  # noqa: E402
import signals as sg  # noqa: E402

ART = REPO / "artifacts"
DETECTOR = config.BM25_VEC            # the detector the v2.5 calibration chose (bm25)
LABEL_K = config.ANSWER_K            # full_gold@ANSWER_K -> answerable from the focused context
K = config.TOP_K                     # signal window

# ---------------------------------------------------------------------------
# Feature engineering: the original 7 candidates + ~12 NEW cheap signals, all
# read off ONE hybrid retrieval result (raw dense cosines, both sparse rankings,
# fused scores). No extra queries, so every signal stays tier-1 cheap.
# ---------------------------------------------------------------------------
ORIG = ["dense_gap", "dense_variance", "confidence_gap", "score_variance",
        "max_score", "evidence_coverage", "divergence_bm25"]
NEW = ["dense_top1", "dense_mean", "dense_margin", "dense_cv", "dense_entropy", "dense_skew",
       "fused_margin", "fused_cv", "fused_entropy", "divergence_rbo_bm25",
       "top1_agree_bm25", "fused_dense_overlap"]
FEATURES = ORIG + NEW


def _entropy(xs: list[float]) -> float:
    """Normalised Shannon entropy (0..1) of a softmax over the scores. Flat ranking
    (retriever cannot separate its candidates) -> high entropy -> weak."""
    if len(xs) < 2:
        return 0.0
    m = max(xs)
    exp = [math.exp(v - m) for v in xs]
    z = sum(exp)
    p = [e / z for e in exp]
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)
    return h / math.log(len(xs))


def _skew(xs: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mu = st.mean(xs)
    sd = st.pstdev(xs)
    if sd == 0:
        return 0.0
    return sum(((x - mu) / sd) ** 3 for x in xs) / len(xs)


def features_from_result(res) -> dict:
    """The 19 candidate signals computed from an EXISTING hybrid retrieval result (no gold,
    no extra query). Shared by the offline feature matrix and the live downstream gate."""
    vals = sg.signal_values(res, detector=DETECTOR, k=K)
    ds = [s for _, s in (res.raw or {}).get("dense", [])][:K]      # raw dense cosines
    fs = list(res.scores)                                          # fused (RRF) scores of top-K
    dense_ids = [i for i, _ in (res.raw or {}).get("dense", [])][:K]
    bm25_ids = [i for i, _ in (res.raw or {}).get(DETECTOR, [])][:K]

    def cv(xs):
        m = st.mean(xs) if xs else 0.0
        return (st.pstdev(xs) / m) if (len(xs) >= 2 and m) else 0.0

    return {
        # --- original 7 ---
        "dense_gap": vals["dense_gap"],
        "dense_variance": vals["dense_variance"],
        "confidence_gap": vals["confidence_gap"],
        "score_variance": vals["score_variance"],
        "max_score": vals["max_score"],
        "evidence_coverage": vals["evidence_coverage"],
        "divergence_bm25": sg.divergence(res.raw, DETECTOR, K, "overlap"),
        # --- new engineered candidates ---
        "dense_top1": ds[0] if ds else 0.0,
        "dense_mean": st.mean(ds) if ds else 0.0,
        "dense_margin": (ds[0] - ds[1]) if len(ds) >= 2 else 0.0,
        "dense_cv": cv(ds),
        "dense_entropy": _entropy(ds),
        "dense_skew": _skew(ds),
        "fused_margin": (fs[0] - fs[1]) if len(fs) >= 2 else 0.0,
        "fused_cv": cv(fs),
        "fused_entropy": _entropy(fs),
        "divergence_rbo_bm25": sg.divergence(res.raw, DETECTOR, K, "rbo"),
        "top1_agree_bm25": 1.0 if (dense_ids and bm25_ids and dense_ids[0] == bm25_ids[0]) else 0.0,
        "fused_dense_overlap": (len(set(res.doc_ids[:K]) & set(dense_ids)) / K) if dense_ids else 0.0,
    }


def ext_feature_row(q: dict) -> dict:
    res = retrieval.search(q["question"], mode="hybrid", k=K, fusion="rrf")
    gold = set(q.get("gold_doc_ids", []))
    fg = 1 if gold and gold.issubset(set(res.doc_ids[:LABEL_K])) else 0
    return {
        "question_id": q["id"],
        "query_type": q["query_type"],
        "n_hops": q.get("n_hops"),
        "full_gold_label": fg,
        **features_from_result(res),
    }


def build_features(split: str, source: str = "frozen") -> list[dict]:
    # frozen = the v2.5 manifest selection (n=480 total); full = ALL answerable-with-gold
    # queries in the split (n=2250 total). Both keep the source-tagged split boundary, so the
    # train/select/test discipline holds either way.
    if source == "full":
        qs = [q for q in data.load_questions_mixed(split) if q.get("answerable") and q.get("gold_doc_ids")]
    else:
        qs = [q for q in data.load_mixed_eval(split) if q.get("answerable") and q.get("gold_doc_ids")]
    return [ext_feature_row(q) for q in qs]


def load_or_build(split: str, recompute: bool, source: str = "frozen") -> list[dict]:
    stem = "features_pool" if source == "full" else "features_ext"
    p = ART / f"{stem}_{split}.json"
    if p.exists() and not recompute:
        rows = json.loads(p.read_text())
        # guard: rebuild if the cache predates a new engineered feature
        if rows and all(f in rows[0] for f in FEATURES):
            print(f"loaded cached features [{source}]: {split} (n={len(rows)})")
            return rows
        print(f"cache for {split} [{source}] missing new features -> rebuilding")
    print(f"building features [{source}] (live retrieval): {split} ...")
    rows = build_features(split, source)
    p.write_text(json.dumps(rows, indent=2))
    print(f"  wrote {p} (n={len(rows)})")
    return rows


def ratio_subsample(rows: list[dict], single_frac: float = 2 / 3) -> list[dict]:
    """Deterministically subsample to the realistic single:multi mix (default 2:1). Single-hop
    is the limiting reagent in the full pool, so keep ALL single-hop and cap multi-hop to match."""
    single = sorted([r for r in rows if r["query_type"] == "single_hop"], key=lambda r: r["question_id"])
    multi = sorted([r for r in rows if r["query_type"] == "multi_hop"], key=lambda r: r["question_id"])
    n_multi = round(len(single) * (1 - single_frac) / single_frac)  # 2:1 -> multi = single/2
    return single + multi[:n_multi]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _y_bad(rows):
    return [1 - r["full_gold_label"] for r in rows]


def single_signal_scores(rows, feat):
    """Oriented so HIGHER == more likely BAD; returns (scores, direction)."""
    from sklearn.metrics import roc_auc_score
    y = _y_bad(rows)
    x = [r[feat] for r in rows]
    pairs = [(a, b) for a, b in zip(y, x) if not (isinstance(b, float) and math.isnan(b))]
    if len({p[0] for p in pairs}) < 2:
        return x, "high=bad"
    a = roc_auc_score([p[0] for p in pairs], [p[1] for p in pairs])
    if a < 0.5:                     # signal is inversely related: low value = bad
        return [-v for v in x], "low=bad"
    return x, "high=bad"


def auc(y, score):
    from sklearn.metrics import roc_auc_score
    pairs = [(a, b) for a, b in zip(y, score) if not (isinstance(b, float) and math.isnan(b))]
    if len({p[0] for p in pairs}) < 2:
        return float("nan")
    return round(float(roc_auc_score([p[0] for p in pairs], [p[1] for p in pairs])), 4)


def ap(y, score):
    from sklearn.metrics import average_precision_score
    pairs = [(a, b) for a, b in zip(y, score) if not (isinstance(b, float) and math.isnan(b))]
    if len({p[0] for p in pairs}) < 2:
        return float("nan")
    return round(float(average_precision_score([p[0] for p in pairs], [p[1] for p in pairs])), 4)


def youden_threshold(y, score):
    """Threshold on `score` (higher=bad) maximising TPR-FPR."""
    pairs = sorted({s for s in score if not (isinstance(s, float) and math.isnan(s))})
    best_t, best_j = None, -1.0
    for t in pairs:
        tp = sum(1 for yi, s in zip(y, score) if s >= t and yi == 1)
        fp = sum(1 for yi, s in zip(y, score) if s >= t and yi == 0)
        fn = sum(1 for yi, s in zip(y, score) if s < t and yi == 1)
        tn = sum(1 for yi, s in zip(y, score) if s < t and yi == 0)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, t
    return best_t


def confusion(y, score, t):
    tp = sum(1 for yi, s in zip(y, score) if s >= t and yi == 1)
    fp = sum(1 for yi, s in zip(y, score) if s >= t and yi == 0)
    fn = sum(1 for yi, s in zip(y, score) if s < t and yi == 1)
    tn = sum(1 for yi, s in zip(y, score) if s < t and yi == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "fpr": round(fpr, 4),
            "f1": round(f1, 4), "accuracy": round((tp + tn) / len(y), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def matrix(rows, feats):
    return [[r[f] for f in feats] for r in rows]


def fit_predict(model_kind, feats, cal, val, test):
    """Fit on calibration, return predicted P(bad) for cal/val/test plus coefficients."""
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    Xc, Xv, Xt = matrix(cal, feats), matrix(val, feats), matrix(test, feats)
    yc = _y_bad(cal)

    if model_kind.startswith("lr"):
        # sklearn 1.8 API: l1_ratio is the knob (1.0 -> L1/sparse, 0.0 -> L2). saga
        # supports both; liblinear+penalty='l1' silently fell back to L2 here.
        if model_kind == "lr_l1":
            clf = LogisticRegression(solver="saga", l1_ratio=1.0, C=1.0, max_iter=5000, class_weight="balanced")
        else:
            clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=5000, class_weight="balanced")
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()), ("clf", clf)])
    else:  # gbt
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                            learning_rate=0.05, subsample=0.8,
                                                            random_state=0))])
    pipe.fit(Xc, yc)

    def pp(X):
        return [round(float(p), 6) for p in pipe.predict_proba(X)[:, 1]]

    # 5-fold CV AUC on calibration (scaling/imputation inside the fold -> no leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_auc = cross_val_score(pipe, Xc, yc, cv=cv, scoring="roc_auc")

    coef = None
    if model_kind.startswith("lr"):
        c = pipe.named_steps["clf"].coef_[0]
        coef = {f: round(float(w), 4) for f, w in zip(feats, c)}  # standardised-feature weights
    else:
        imp = pipe.named_steps["clf"].feature_importances_
        coef = {f: round(float(w), 4) for f, w in zip(feats, imp)}

    return {"pc": pp(Xc), "pv": pp(Xv), "pt": pp(Xt),
            "cv_auc_mean": round(float(np.mean(cv_auc)), 4),
            "cv_auc_std": round(float(np.std(cv_auc)), 4),
            "coef": coef}


def by_type_auc(rows, score):
    out = {}
    for qt in ("single_hop", "multi_hop"):
        idx = [i for i, r in enumerate(rows) if r["query_type"] == qt]
        y = [1 - rows[i]["full_gold_label"] for i in idx]
        s = [score[i] for i in idx]
        out[qt] = {"n": len(idx), "auc": auc(y, s)}
    return out


def evaluate(name, sc_cal, sc_val, sc_test, cal, val, test, extra=None):
    yc, yv, yt = _y_bad(cal), _y_bad(val), _y_bad(test)
    t = youden_threshold(yc, sc_cal)            # operating point chosen on calibration only
    rec = {
        "name": name,
        "auc_cal": auc(yc, sc_cal), "auc_val": auc(yv, sc_val), "auc_test": auc(yt, sc_test),
        "ap_val": ap(yv, sc_val), "ap_test": ap(yt, sc_test),
        "by_type_val": by_type_auc(val, sc_val),
        "op_val": confusion(yv, sc_val, t), "op_test": confusion(yt, sc_test, t),
    }
    if extra:
        rec.update(extra)
    return rec


def operating_at_recall(yc, sc_c, yt, sc_t, target):
    """Pick the threshold on CALIBRATION that just meets `target` recall (the highest such
    threshold -> lowest FPR), read it out on TEST. Apples-to-apples: same catch rate, compare
    wasted escalation (FPR). Returns the test confusion plus the realised cal/test recall."""
    npos = sum(yc)
    if not npos:
        return None
    chosen = min(sc_c)
    for t in sorted({s for s in sc_c}, reverse=True):
        rec = sum(1 for y, s in zip(yc, sc_c) if s >= t and y == 1) / npos
        if rec >= target:
            chosen = t
            break
    conf = confusion(yt, sc_t, chosen)
    cal_rec = round(sum(1 for y, s in zip(yc, sc_c) if s >= chosen and y == 1) / npos, 4)
    conf["target_recall"] = target
    conf["cal_recall"] = cal_rec
    return conf


def or_gate_operating(rows, thr):
    """The LIVE v2.5 gate: bad if ANY selected weakness signal fires below its floor."""
    sel = thr.get("_weakness_signals") or ["dense_variance", "score_variance"]
    y = _y_bad(rows)
    pred = []
    for r in rows:
        fired = any(r.get(s, float("inf")) < thr[s] for s in sel)
        pred.append(1 if fired else 0)
    tp = sum(1 for yi, p in zip(y, pred) if p and yi == 1)
    fp = sum(1 for yi, p in zip(y, pred) if p and yi == 0)
    fn = sum(1 for yi, p in zip(y, pred) if not p and yi == 1)
    tn = sum(1 for yi, p in zip(y, pred) if not p and yi == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "fpr": round(fpr, 4),
            "f1": round(f1, 4), "accuracy": round((tp + tn) / len(y), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "selected": sel}


# ---------------------------------------------------------------------------
def run_study(cal, val, test, thr, stem: str, scenario: str) -> dict:
    base = {
        "all": round(sum(_y_bad(val)) / len(val), 3),
        "single_hop": round(sum(1 - r["full_gold_label"] for r in val if r["query_type"] == "single_hop")
                            / max(1, sum(1 for r in val if r["query_type"] == "single_hop")), 3),
        "multi_hop": round(sum(1 - r["full_gold_label"] for r in val if r["query_type"] == "multi_hop")
                           / max(1, sum(1 for r in val if r["query_type"] == "multi_hop")), 3),
    }

    # --- 1. single-signal ranking (every candidate, old + new), by validation AUC ---
    yv = _y_bad(val)
    signal_rank = []
    for f in FEATURES:
        sc_v, direction = single_signal_scores(val, f)
        sc_c, _ = single_signal_scores(cal, f)
        sc_t, _ = single_signal_scores(test, f)
        signal_rank.append({
            "signal": f, "is_new": f in NEW, "direction": direction,
            "auc_cal": auc(_y_bad(cal), sc_c), "auc_val": auc(yv, sc_v), "auc_test": auc(_y_bad(test), sc_t),
            "by_type_val": by_type_auc(val, sc_v),
        })
    signal_rank.sort(key=lambda d: -(d["auc_val"] if d["auc_val"] == d["auc_val"] else 0))

    # --- 2. model lineup ---
    models = []
    scored = {}   # name -> (cal_scores, test_scores) for matched-recall operating analysis
    yc, yt = _y_bad(cal), _y_bad(test)

    # single-signal baselines (evaluated as scores, oriented high=bad)
    for f in ("dense_variance", "score_variance"):
        scv, _ = single_signal_scores(val, f)
        scc, _ = single_signal_scores(cal, f)
        sct, _ = single_signal_scores(test, f)
        models.append(evaluate(f"single:{f}", scc, scv, sct, cal, val, test,
                               extra={"kind": "single", "features": [f]}))
        scored[f"single:{f}"] = (scc, sct)
    # best single overall (top of the ranking)
    best_sig = signal_rank[0]["signal"]
    if best_sig not in ("dense_variance", "score_variance"):
        scv, _ = single_signal_scores(val, best_sig)
        scc, _ = single_signal_scores(cal, best_sig)
        sct, _ = single_signal_scores(test, best_sig)
        models.append(evaluate(f"single:{best_sig}(best)", scc, scv, sct, cal, val, test,
                               extra={"kind": "single", "features": [best_sig]}))

    feat_sets = {
        "lr_l2:selected2": ("lr_l2", ["dense_variance", "score_variance"]),
        "lr_l2:orig7": ("lr_l2", ORIG),
        "lr_l2:ext19": ("lr_l2", FEATURES),
        "lr_l1:ext19": ("lr_l1", FEATURES),
        "gbt:ext19": ("gbt", FEATURES),
    }
    for name, (kind, feats) in feat_sets.items():
        r = fit_predict(kind, feats, cal, val, test)
        models.append(evaluate(name, r["pc"], r["pv"], r["pt"], cal, val, test,
                               extra={"kind": kind, "features": feats, "coef": r["coef"],
                                      "cv_auc_mean": r["cv_auc_mean"], "cv_auc_std": r["cv_auc_std"]}))
        scored[name] = (r["pc"], r["pt"])

    # --- 3. operating-point ---
    # 3a. the LIVE OR-gate (a single fixed point; no tunable threshold)
    or_gate = {"val": or_gate_operating(val, thr), "test": or_gate_operating(test, thr)}
    # 3b. matched-recall: at the SAME catch rate, which gate wastes less escalation (FPR)?
    #     thresholds set on calibration, read on test. Anchored near the live OR-gate's recall.
    matched = {}
    for tgt in (0.80, 0.90):
        row = {}
        for name in ("single:dense_variance", "lr_l2:selected2", "lr_l2:ext19", "lr_l1:ext19"):
            sc_c, sc_t = scored[name]
            row[name] = operating_at_recall(yc, sc_c, yt, sc_t, tgt)
        matched[f"recall_{int(tgt*100)}"] = row

    out = {
        "label": f"full_gold@{LABEL_K}",
        "scenario": scenario,
        "splits": {"calibration": len(cal), "validation": len(val), "test": len(test)},
        "base_rate_bad_val": base,
        "detector": DETECTOR,
        "n_features": {"original": len(ORIG), "new": len(NEW), "total": len(FEATURES)},
        "signal_ranking": signal_rank,
        "models": models,
        "or_gate": or_gate,
        "matched_recall": matched,
        "best_single_overall": best_sig,
    }
    (ART / f"{stem}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{scenario}]  n = {len(cal)}/{len(val)}/{len(test)} (cal/val/test)")
    print(f"wrote artifacts/{stem}.json")

    # console summary
    print(f"\n=== single-signal ranking by validation AUC (detect {out['label']}==0) ===")
    print(f"{'signal':24s} {'new':>4s} {'cal':>7s} {'val':>7s} {'test':>7s} {'val/1hop':>9s} {'val/multi':>10s}")
    for s in signal_rank:
        bt = s["by_type_val"]
        print(f"{s['signal']:24s} {'*' if s['is_new'] else '':>4s} "
              f"{s['auc_cal']:7.3f} {s['auc_val']:7.3f} {s['auc_test']:7.3f} "
              f"{bt['single_hop']['auc']:9.3f} {bt['multi_hop']['auc']:10.3f}")
    print(f"\n=== model AUC (cal / val / test) + PR-AUC(val) + CV ===")
    print(f"{'model':22s} {'cal':>7s} {'val':>7s} {'test':>7s} {'AP/val':>7s} {'cv_auc':>13s}")
    for m in models:
        cvs = f"{m.get('cv_auc_mean','-')}" + (f"±{m['cv_auc_std']}" if 'cv_auc_std' in m else "")
        print(f"{m['name']:22s} {m['auc_cal']:7.3f} {m['auc_val']:7.3f} {m['auc_test']:7.3f} "
              f"{m['ap_val']:7.3f} {cvs:>13s}")
    print(f"building {stem}.html ...")
    build_html(out, REPO / f"{stem}.html")
    print(f"wrote {stem}.html")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--recompute", action="store_true")
    ap_.add_argument("--pool", choices=["frozen", "full"], default="frozen",
                     help="frozen = v2.5 manifest (n=480); full = all answerable-with-gold (n=2250)")
    args = ap_.parse_args()
    thr = json.loads((ART / "thresholds_mixed.json").read_text())

    if args.pool == "frozen":
        cal = load_or_build("calibration", args.recompute, "frozen")
        val = load_or_build("validation", args.recompute, "frozen")
        test = load_or_build("test", args.recompute, "frozen")
        run_study(cal, val, test, thr, "classifier_study", "frozen v2.5 population")
    else:
        cal = load_or_build("calibration", args.recompute, "full")
        val = load_or_build("validation", args.recompute, "full")
        test = load_or_build("test", args.recompute, "full")
        run_study(cal, val, test, thr, "classifier_study_full", "full available pool")
        # ratio-preserved 2:1 single:multi, subsampled from the SAME matrices (no extra retrieval)
        rcal, rval, rtest = ratio_subsample(cal), ratio_subsample(val), ratio_subsample(test)
        run_study(rcal, rval, rtest, thr, "classifier_study_ratio", "ratio-preserved 2:1 single:multi")
    return 0


# ---------------------------------------------------------------------------
# HTML report (styled to match briefing.html)
# ---------------------------------------------------------------------------
def build_html(d: dict, out_path) -> None:
    import html as _h

    def f3(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "&mdash;"

    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) and x == x else "&mdash;"

    def delta(x, ref):
        if not (isinstance(x, (int, float)) and isinstance(ref, (int, float))):
            return ""
        dv = x - ref
        cls = "pos" if dv > 0.0005 else ("neg" if dv < -0.0005 else "flat")
        return f' <span class="{cls}">({"+" if dv >= 0 else ""}{dv:.3f})</span>'

    models = {m["name"]: m for m in d["models"]}
    base_dv = models.get("single:dense_variance", {})
    ref_val = base_dv.get("auc_val", float("nan"))
    ref_test = base_dv.get("auc_test", float("nan"))

    # signal ranking rows
    sig_rows = ""
    for s in d["signal_ranking"]:
        bt = s["by_type_val"]
        tag = '<span class="badge badge-amber">new</span>' if s["is_new"] else ''
        sig_rows += (f"<tr><td style='text-align:left'>{_h.escape(s['signal'])} {tag}</td>"
                     f"<td>{_h.escape(s['direction'])}</td>"
                     f"<td>{f3(s['auc_cal'])}</td><td><b>{f3(s['auc_val'])}</b></td><td>{f3(s['auc_test'])}</td>"
                     f"<td>{f3(bt['single_hop']['auc'])}</td><td>{f3(bt['multi_hop']['auc'])}</td></tr>")

    # model rows
    mod_rows = ""
    order = ["single:dense_variance", "single:score_variance"]
    order += [m["name"] for m in d["models"] if m["name"].endswith("(best)")]
    order += ["lr_l2:selected2", "lr_l2:orig7", "lr_l2:ext19", "lr_l1:ext19", "gbt:ext19"]
    for name in order:
        m = models.get(name)
        if not m:
            continue
        bt = m["by_type_val"]
        cvs = (f"{m['cv_auc_mean']:.3f}±{m['cv_auc_std']:.3f}" if "cv_auc_mean" in m else "&mdash;")
        hi = ' style="background:#fff6fa"' if name == "lr_l2:selected2" else ""
        mod_rows += (f"<tr{hi}><td style='text-align:left'>{_h.escape(name)}</td>"
                     f"<td>{f3(m['auc_cal'])}</td>"
                     f"<td><b>{f3(m['auc_val'])}</b>{delta(m['auc_val'], ref_val)}</td>"
                     f"<td>{f3(m['auc_test'])}{delta(m['auc_test'], ref_test)}</td>"
                     f"<td>{f3(m['ap_val'])}</td>"
                     f"<td>{f3(bt['single_hop']['auc'])}</td><td>{f3(bt['multi_hop']['auc'])}</td>"
                     f"<td>{cvs}</td></tr>")

    # operating-point rows (val + test) vs OR-gate
    def op_row(label, op, hi=False):
        style = ' style="background:#fff6fa"' if hi else ""
        return (f"<tr{style}><td style='text-align:left'>{_h.escape(label)}</td>"
                f"<td>{pct(op['recall'])}</td><td>{pct(op['precision'])}</td>"
                f"<td>{pct(op['fpr'])}</td><td>{pct(op['f1'])}</td><td>{pct(op['accuracy'])}</td></tr>")

    op_rows = op_row("OR-gate (live v2.5: dense_variance OR score_variance)", d["or_gate"]["test"])
    for name in ["single:dense_variance", "lr_l2:selected2", "lr_l2:ext19", "lr_l1:ext19", "gbt:ext19"]:
        m = models.get(name)
        if m:
            op_rows += op_row(f"{name}  @cal-Youden", m["op_test"], hi=(name == "lr_l2:selected2"))

    # matched-recall table: at the SAME catch rate, who wastes less escalation?
    PRETTY = {"single:dense_variance": "best single (dense_variance)",
              "lr_l2:selected2": "LR (dense_variance + score_variance)",
              "lr_l2:ext19": "LR (all 19)", "lr_l1:ext19": "LR-L1 (all 19, sparse)"}
    matched_rows = ""
    for tgt_key in ("recall_90", "recall_80"):
        block = d.get("matched_recall", {}).get(tgt_key, {})
        tgt = tgt_key.split("_")[1]
        for name in ("single:dense_variance", "lr_l2:selected2", "lr_l2:ext19", "lr_l1:ext19"):
            o = block.get(name)
            if not o:
                continue
            hi = ' style="background:#fff6fa"' if name == "lr_l2:selected2" else ""
            matched_rows += (f"<tr{hi}><td style='text-align:left'>~{tgt}% &middot; {_h.escape(PRETTY[name])}</td>"
                             f"<td>{pct(o['recall'])}</td><td>{pct(o['fpr'])}</td>"
                             f"<td>{pct(o['precision'])}</td></tr>")

    # LR coefficients (selected2 + ext L1)
    def coef_block(name):
        m = models.get(name, {})
        coef = m.get("coef", {})
        if not coef:
            return ""
        items = sorted(coef.items(), key=lambda kv: -abs(kv[1]))
        rows = "".join(f"<tr><td style='text-align:left'>{_h.escape(k)}</td><td>{v:+.3f}</td></tr>"
                       for k, v in items if abs(v) > 1e-6)
        return (f"<h3>{_h.escape(name)} weights (standardised features)</h3>"
                f"<table><tr><th>feature</th><th>weight</th></tr>{rows}</table>")

    coefs = coef_block("lr_l2:selected2") + coef_block("lr_l1:ext19")

    base = d["base_rate_bad_val"]
    best = d["best_single_overall"]
    # headline numbers for the TL;DR
    sel2 = models.get("lr_l2:selected2", {})
    ext = models.get("lr_l2:ext19", {})
    # matched-recall references (so the prose can't drift from the table)
    mr = d.get("matched_recall", {})
    m80 = mr.get("recall_80", {})
    m80_single = (m80.get("single:dense_variance") or {}).get("fpr")
    m80_lr = (m80.get("lr_l2:selected2") or {}).get("fpr")
    og_test = d["or_gate"]["test"]

    # by-type validation AUC (for the weighted-compromise takeaway) - all data-driven
    def bt(name, qt):
        return (models.get(name, {}).get("by_type_val", {}).get(qt, {}) or {}).get("auc")
    dv_1, dv_m = bt("single:dense_variance", "single_hop"), bt("single:dense_variance", "multi_hop")
    sv_1, sv_m = bt("single:score_variance", "single_hop"), bt("single:score_variance", "multi_hop")
    lr_1, lr_m = bt("lr_l2:selected2", "single_hop"), bt("lr_l2:selected2", "multi_hop")
    # strongest ENGINEERED signal on the multi-hop slice
    new_sigs = [s for s in d["signal_ranking"] if s["is_new"]]
    best_new_multi = max(new_sigs, key=lambda s: (s["by_type_val"]["multi_hop"]["auc"]
                         if s["by_type_val"]["multi_hop"]["auc"] == s["by_type_val"]["multi_hop"]["auc"] else 0),
                         default=None) if new_sigs else None
    bnm_name = best_new_multi["signal"] if best_new_multi else "&mdash;"
    bnm_auc = best_new_multi["by_type_val"]["multi_hop"]["auc"] if best_new_multi else None

    css = """:root{--ink:#1a1a22;--muted:#5b6270;--line:#e5e7eb;--bg:#fafafb;--card:#fff;--accent:#d6336c;--accent2:#1c7ed6;--pos:#138a52;--neg:#c92a2a;--amber:#b8860b;}
*{box-sizing:border-box;}body{margin:0;font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);}
.wrap{max-width:1080px;margin:0 auto;padding:40px 48px 100px;}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;}h2{font-size:23px;margin:44px 0 14px;padding-top:10px;border-top:1px solid var(--line);}
h3{font-size:16px;margin:22px 0 8px;}p,li{color:#23262e;}.sub{color:var(--muted);font-size:14px;}
.tag{display:inline-block;background:#fdeef4;color:var(--accent);border:1px solid #f6c6da;border-radius:999px;padding:2px 11px;font-size:12.5px;font-weight:600;margin-right:6px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.03);}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}.kpi{text-align:center;}.kpi .n{font-size:21px;font-weight:700;}.kpi .l{font-size:12.5px;color:var(--muted);}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13.5px;}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top;}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);}td:not(:first-child),th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums;}
.pos{color:var(--pos);font-weight:600;}.neg{color:var(--neg);font-weight:600;}.flat{color:var(--muted);}
.badge{font-size:10.5px;padding:1px 7px;border-radius:6px;font-weight:600;}.badge-amber{background:#fdf3e0;color:var(--amber);}
.callout{border-left:4px solid var(--accent);background:#fff;padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0;}.callout.good{border-color:var(--pos);}.callout.warn{border-color:var(--amber);}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:#f2f3f5;padding:1px 5px;border-radius:4px;}
ul.tight li{margin:5px 0;}.foot{color:var(--muted);font-size:13px;margin-top:50px;border-top:1px solid var(--line);padding-top:16px;}"""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Classifier signal-fusion study</title><style>{css}</style></head><body><div class="wrap">
<div><span class="tag">Branch study</span><span class="tag">classifier-signal-fusion</span><span class="tag">{_h.escape(d.get('scenario',''))}</span><span class="tag">n={d['splits']['calibration']}/{d['splits']['validation']}/{d['splits']['test']}</span></div>
<h1>Can a classifier fuse the signals better than one threshold?</h1>
<p class="sub">The CTO's question, run with the v2.5 eval discipline: detect weak retrieval
(<span class="mono">full_gold@{d['label'].split('@')[1]}==0</span>), TRAIN on calibration
(n={d['splits']['calibration']}), SELECT on validation (n={d['splits']['validation']}),
REPORT once on test (n={d['splits']['test']}). Detector: {d['detector']}. Higher AUC = better
separation of good vs weak retrieval. Baseline to beat = the best single signal
(<span class="mono">dense_variance</span>).</p>

<div class="card grid4">
  <div class="kpi"><div class="n">{f3(ref_val)} &rarr; {f3(sel2.get('auc_val'))}</div><div class="l">val AUC: best single &rarr; LR(dense_variance + score_variance)</div></div>
  <div class="kpi"><div class="n">{f3(ref_test)} &rarr; {f3(sel2.get('auc_test'))}</div><div class="l">test AUC: same comparison (held out)</div></div>
  <div class="kpi"><div class="n">{d['n_features']['new']}</div><div class="l">new candidate signals engineered &amp; tested</div></div>
  <div class="kpi"><div class="n">{f3(ext.get('auc_cal'))} / {f3(ext.get('auc_test'))}</div><div class="l">LR(all 19) cal vs test AUC (overfit gap)</div></div>
</div>

<h2>1. The model lineup</h2>
<p>Single-signal thresholds vs trained fusion gates. <b>val</b> is the selection number;
<b>test</b> is touched once. Deltas in parentheses are vs the best single signal
(<span class="mono">dense_variance</span>). by-type AUC is on validation. <span class="mono">cv_auc</span>
is 5-fold on calibration (stability / overfit check). The highlighted row is the interpretable
minimal-fusion recommendation.</p>
<table>
<tr><th>model</th><th>AUC cal</th><th>AUC val</th><th>AUC test</th><th>PR-AUC val</th><th>val single-hop</th><th>val multi-hop</th><th>cv AUC (cal)</th></tr>
{mod_rows}
</table>

<h2>2. Hunting new signals</h2>
<p>Every candidate signal, old and <span class="badge badge-amber">new</span>, ranked by
validation AUC. Direction shows whether high or low values flag weak retrieval. This is the
"test it on your own corpus" methodology applied to {d['n_features']['total']} signals at once.</p>
<table>
<tr><th>signal</th><th>direction</th><th>AUC cal</th><th>AUC val</th><th>AUC test</th><th>val single-hop</th><th>val multi-hop</th></tr>
{sig_rows}
</table>
<p class="sub">Base rate of weak retrieval on validation: all {pct(base['all'])} &middot;
single-hop {pct(base['single_hop'])} &middot; multi-hop {pct(base['multi_hop'])}. Single-hop
retrieval is mostly strong, so weak cases are rare there; the multi-hop slice is where detection
has the most to separate.</p>

<h2>3. At a real operating point</h2>
<p>AUC is ranking quality; the gate ships a threshold. <b>recall</b> = of truly-weak retrievals,
how many we catch (and escalate); <b>FPR</b> = of strong retrievals, how many we needlessly
escalate (wasted cost); <b>precision</b> = of what we escalate, how much truly needed it.</p>

<h3>3a. Each gate's own max-Youden point (NOT comparable across rows)</h3>
<p>Threshold chosen on calibration by Youden J, read on test. The gates land at different recall
levels, so do not read this as "LR catches less" - that is just a different threshold. Use 3b for a
fair comparison.</p>
<table>
<tr><th>gate (on test)</th><th>recall (catch weak)</th><th>precision</th><th>FPR (waste)</th><th>F1</th><th>accuracy</th></tr>
{op_rows}
</table>

<h3>3b. Matched recall: same catch rate, who wastes less? (the fair comparison)</h3>
<p>Threshold set on calibration to hit a target catch rate, read on test. At the SAME recall, a
lower FPR means less strong retrieval needlessly escalated - real saved cost. The live OR-gate sits
near 90% recall at {pct(d['or_gate']['test']['fpr'])} FPR (row above) for reference.</p>
<table>
<tr><th>target recall &middot; gate</th><th>test recall</th><th>FPR (waste)</th><th>precision</th></tr>
{matched_rows}
</table>

<h2>4. What the model weighs</h2>
<p>Standardised-feature weights (logistic regression) and split importances. This is the
interpretability the workshop story needs: a fused gate you can still read.</p>
{coefs}

<h2>Takeaways</h2>
<ul class="tight">
<li><b>The fusion lift shows up in AUC but mostly vanishes at a real operating point.</b>
LR(dense_variance + score_variance) is the best model on ranking quality (test AUC {f3(ref_test)} &rarr;
{f3(sel2.get('auc_test'))}, PR-AUC {f3(base_dv.get('ap_val'))} &rarr; {f3(sel2.get('ap_val'))}). But at
matched recall (section 3b) it ties the single signal: at ~80% catch rate both waste {pct(m80_lr)} of
strong retrieval ({pct(m80_single)} for the single signal). The AUC gain (+{(sel2.get('auc_test',0)-ref_test):.3f})
is also inside the 5-fold CV spread (&plusmn;{sel2.get('cv_auc_std','?')}). Net: marginal and
corpus-specific, not a step change.</li>
<li><b>The cheapest win here is not the classifier at all.</b> The live OR-gate catches
{pct(og_test['recall'])} of weak retrieval at {pct(og_test['fpr'])} FPR; a single well-set
dense_variance threshold hits the same catch rate at {pct(m80_single)} FPR. Firing when EITHER signal
trips over-escalates - tightening the gate to one calibrated signal saves wasted cost with no model at
all. Worth checking independently of this study.</li>
<li><b>More signals and fancier models do NOT help - they overfit.</b> Adding all 19 features or
switching to gradient boosting lowers test AUC and blows the cal&rarr;test gap wide open (GBT:
cal {f3(models.get('gbt:ext19',{}).get('auc_cal'))} vs test {f3(models.get('gbt:ext19',{}).get('auc_test'))}
on {d['splits']['calibration']} calibration rows). On data this size, two regularised signals beat
nineteen.</li>
<li><b>Fusion is a weighted compromise across query types.</b> On validation (single-hop / multi-hop AUC):
dense_variance {f3(dv_1)} / {f3(dv_m)}; score_variance {f3(sv_1)} / {f3(sv_m)}; the fused LR
{f3(lr_1)} / {f3(lr_m)}. The fused gate tracks the stronger signal on each type but is pulled toward the
majority class, so a type-specific signal can still beat it on the minority slice. If one query type is
what you care about, its dedicated signal may still win there.</li>
<li><b>The strongest single signal overall is still {_h.escape(d.get('best_single_overall',''))}.</b>
None of the 12 engineered signals top it on overall validation AUC. The most interesting of the new
ones is {_h.escape(bnm_name)}, the strongest engineered signal on the multi-hop slice (val multi-hop AUC
{f3(bnm_auc)}) - a candidate for the "test on your corpus" catalog, not the notebook spine.</li>
<li><b>Interpretability is preserved.</b> The winning gate is two readable weights
(dense_variance {models.get('lr_l2:selected2',{}).get('coef',{}).get('dense_variance','?')},
score_variance {models.get('lr_l2:selected2',{}).get('coef',{}).get('score_variance','?')}; both
negative = low value flags weak retrieval). Adopting it would not turn the workshop into an
ML-modelling lesson.</li>
<li><b>This measures DETECTION only.</b> Whether the marginal AUC/PR-AUC lift actually moves the
end-to-end cost/quality frontier needs the classifier dropped into the live gate and a headline
re-run. Given the size of the lift here, do that before committing to it in the notebook.</li>
</ul>
<div class="foot">Generated by <span class="mono">scripts/run_classifier_study.py</span>
(scenario: {_h.escape(d.get('scenario',''))}, n={d['splits']['calibration']}/{d['splits']['validation']}/{d['splits']['test']} cal/val/test).
Numbers in <span class="mono">artifacts/{_h.escape(out_path.stem)}.json</span>.</div>
</div></body></html>"""
    out_path.write_text(doc)


if __name__ == "__main__":
    sys.exit(main())
