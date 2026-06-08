"""Diagnostic: WHY doesn't the fused classifier beat the single signal?

Runs entirely on the cached feature matrices (no retrieval). Tests the hypotheses for the
null result, and adds the rigorous tests the headline study skipped:

  A. Feature quality      - any degenerate / constant / NaN-heavy features?
  B. Redundancy           - pairwise correlation + PCA effective dimensionality
  C. Complementarity      - greedy forward selection: does adding signals past #1 help?
  D. Achievable ceiling   - CV AUC of strong models (RF/GBT) vs single: is there headroom at all?
  E. Significance         - bootstrap 95% CI of the test AUC DIFFERENCE (single vs fused): excludes 0?
  F. Per-type headroom    - within multi-hop alone, does fusion beat the best single signal?
  G. query_type feature   - does telling the model the hop-type help? (availability caveat)

Usage:
  python scripts/diagnose_classifier.py [--pool full|frozen]   (default full = most power)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import run_classifier_study as rcs  # noqa: E402  (reuse FEATURES/ORIG/NEW + helpers)

FEATURES, ORIG, NEW = rcs.FEATURES, rcs.ORIG, rcs.NEW


def load(pool, split):
    stem = "features_pool" if pool == "full" else "features_ext"
    return json.loads((ART / f"{stem}_{split}.json").read_text())


def y_bad(rows):
    return [1 - r["full_gold_label"] for r in rows]


def Xy(rows, feats):
    import numpy as np
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    return X, np.array(y_bad(rows))


def lr_pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=5000, class_weight="balanced"))])


def auc_of(y, s):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    y = np.asarray(y); s = np.asarray(s)
    m = ~np.isnan(s)
    if len(set(y[m].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["full", "frozen"], default="full")
    args = ap.parse_args()
    import numpy as np

    cal, val, test = load(args.pool, "calibration"), load(args.pool, "validation"), load(args.pool, "test")
    n = (len(cal), len(val), len(test))
    print(f"\n{'='*72}\nDIAGNOSTIC  [pool={args.pool}]  n(cal/val/test) = {n[0]}/{n[1]}/{n[2]}\n{'='*72}")
    out = {"pool": args.pool, "n": {"cal": n[0], "val": n[1], "test": n[2]}}

    # oriented single-signal scores (low dense_variance = weak -> negate so high=weak)
    def single_scores(rows, f):
        return rcs.single_signal_scores(rows, f)[0]

    # ---------- A. feature quality ----------
    print("\n[A] FEATURE QUALITY (on calibration)")
    print(f"  {'feature':22s} {'NaN%':>6s} {'std':>9s} {'n_uniq':>7s}  flag")
    qual = {}
    for f in FEATURES:
        v = np.array([r[f] for r in cal], dtype=float)
        nanpct = float(np.mean(np.isnan(v)) * 100)
        vv = v[~np.isnan(v)]
        std = float(np.std(vv)) if len(vv) else 0.0
        nuniq = int(len(np.unique(vv)))
        flag = "DEGENERATE" if (std == 0 or nuniq <= 2) else ("low-var" if std < 1e-6 else "")
        qual[f] = {"nan_pct": round(nanpct, 1), "std": round(std, 5), "n_unique": nuniq, "flag": flag}
        print(f"  {f:22s} {nanpct:6.1f} {std:9.4f} {nuniq:7d}  {flag}")
    out["feature_quality"] = qual

    # ---------- B. redundancy ----------
    print("\n[B] REDUNDANCY")
    M = np.array([[r[f] for f in FEATURES] for r in cal], dtype=float)
    M = np.where(np.isnan(M), np.nanmedian(M, axis=0), M)
    C = np.corrcoef(M, rowvar=False)
    pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            pairs.append((abs(C[i, j]), FEATURES[i], FEATURES[j], C[i, j]))
    pairs.sort(reverse=True)
    print("  top correlated pairs:")
    for a, fi, fj, c in pairs[:6]:
        print(f"    {c:+.3f}  {fi} ~ {fj}")
    # PCA effective dimensionality on standardized features
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Z = StandardScaler().fit_transform(M)
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    d95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"  PCA effective dim of {len(FEATURES)} features: {d90} comps -> 90% var, {d95} comps -> 95% var")
    print(f"  PC1 alone explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    out["redundancy"] = {"top_pairs": [[round(c, 3), fi, fj] for _, fi, fj, c in pairs[:6]],
                         "pca_dim_90": d90, "pca_dim_95": d95,
                         "pc1_var": round(float(pca.explained_variance_ratio_[0]), 3)}

    # ---------- C. complementarity: greedy forward selection (fit cal, score val) ----------
    print("\n[C] COMPLEMENTARITY - greedy forward selection (fit cal, AUC on val)")
    remaining = list(FEATURES)
    chosen, curve = [], []
    while remaining and len(chosen) < 8:
        best_f, best_auc = None, -1
        for f in remaining:
            feats = chosen + [f]
            pipe = lr_pipe(); pipe.fit(*Xy(cal, feats))
            a = auc_of(y_bad(val), pipe.predict_proba(Xy(val, feats)[0])[:, 1])
            if a > best_auc:
                best_auc, best_f = a, f
        gain = best_auc - (curve[-1][1] if curve else 0.5)
        curve.append((best_f, best_auc, gain))
        chosen.append(best_f); remaining.remove(best_f)
        marker = "  <- plateau" if (len(curve) > 1 and gain < 0.003) else ""
        print(f"  +{best_f:22s} val AUC = {best_auc:.4f}  (gain {gain:+.4f}){marker}")
    out["forward_selection"] = [[f, round(a, 4), round(g, 4)] for f, a, g in curve]

    # ---------- D. achievable ceiling: CV AUC on cal+val (test untouched) ----------
    print("\n[D] ACHIEVABLE CEILING - 5-fold CV AUC on cal+val (is there ANY headroom?)")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pool = cal + val
    yb = np.array(y_bad(pool))
    # single signal: CV AUC = mean over folds of raw-signal AUC on the held fold
    sdv = np.array(single_scores(pool, "dense_variance"))
    s_aucs = []
    for _, te in cv5.split(sdv.reshape(-1, 1), yb):
        s_aucs.append(auc_of(yb[te], sdv[te]))
    print(f"  {'single dense_variance':28s} {np.mean(s_aucs):.4f} ± {np.std(s_aucs):.4f}")
    ceil = {"single_dense_variance": [round(float(np.mean(s_aucs)), 4), round(float(np.std(s_aucs)), 4)]}
    models = {
        "LR(2 selected)": (lr_pipe(), ["dense_variance", "score_variance"]),
        "LR(all 19)": (lr_pipe(), FEATURES),
        "GBT(all 19)": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                                     learning_rate=0.05, subsample=0.8, random_state=0))]), FEATURES),
        "RandomForest(all 19)": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                           ("clf", RandomForestClassifier(n_estimators=400, max_depth=None,
                                                                          min_samples_leaf=5, random_state=0, class_weight="balanced"))]), FEATURES),
    }
    for name, (pipe, feats) in models.items():
        X = np.array([[r[f] for f in feats] for r in pool], dtype=float)
        sc = cross_val_score(pipe, X, yb, cv=cv5, scoring="roc_auc")
        print(f"  {name:28s} {sc.mean():.4f} ± {sc.std():.4f}   (vs single {sc.mean()-np.mean(s_aucs):+.4f})")
        ceil[name] = [round(float(sc.mean()), 4), round(float(sc.std()), 4)]
    out["ceiling_cv"] = ceil

    # ---------- E. significance: bootstrap test AUC difference ----------
    print("\n[E] SIGNIFICANCE - bootstrap 95% CI of TEST AUC difference (fit on cal)")
    rng = np.random.default_rng(0)
    yt = np.array(y_bad(test))
    s_single = np.array(single_scores(test, "dense_variance"))
    fitted = {}
    for name, feats in (("LR(2)", ["dense_variance", "score_variance"]), ("LR(19)", FEATURES)):
        p = lr_pipe(); p.fit(*Xy(cal, feats))
        fitted[name] = p.predict_proba(Xy(test, feats)[0])[:, 1]
    B = 2000
    idx = np.arange(len(yt))
    diffs = {k: [] for k in fitted}
    aucs = {"single": [], **{k: [] for k in fitted}}
    for _ in range(B):
        bs = rng.choice(idx, size=len(idx), replace=True)
        if len(set(yt[bs].tolist())) < 2:
            continue
        a_s = auc_of(yt[bs], s_single[bs]); aucs["single"].append(a_s)
        for k in fitted:
            a_m = auc_of(yt[bs], fitted[k][bs]); aucs[k].append(a_m); diffs[k].append(a_m - a_s)

    def ci(x):
        return (round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4))
    print(f"  point AUC (full test):  single={auc_of(yt, s_single):.4f}  "
          f"LR(2)={auc_of(yt, fitted['LR(2)']):.4f}  LR(19)={auc_of(yt, fitted['LR(19)']):.4f}")
    sig = {}
    for k in fitted:
        lo, hi = ci(diffs[k])
        verdict = "SIGNIFICANT (excludes 0)" if lo > 0 else "NOT significant (CI includes 0)"
        print(f"  {k} - single:  median {np.median(diffs[k]):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
        sig[k] = {"median_diff": round(float(np.median(diffs[k])), 4), "ci95": [lo, hi], "significant": lo > 0}
    out["significance"] = sig

    # ---------- F. per-type headroom (multi-hop is where detection is hard) ----------
    print("\n[F] PER-TYPE HEADROOM - fit & eval WITHIN each query type")
    perF = {}
    for qt in ("single_hop", "multi_hop"):
        c = [r for r in cal if r["query_type"] == qt]
        v = [r for r in val if r["query_type"] == qt]
        if len(set(y_bad(v))) < 2 or len(set(y_bad(c))) < 2:
            print(f"  {qt}: too few of one class to model"); continue
        # best single within type (on val)
        bests = max(FEATURES, key=lambda f: (auc_of(y_bad(v), single_scores(v, f)) or 0))
        a_single = auc_of(y_bad(v), single_scores(v, bests))
        p = lr_pipe(); p.fit(*Xy(c, FEATURES))
        a_lr = auc_of(y_bad(v), p.predict_proba(Xy(v, FEATURES)[0])[:, 1])
        print(f"  {qt:11s}: best single ({bests}) val AUC {a_single:.4f}  |  LR(19) {a_lr:.4f}  (Δ {a_lr-a_single:+.4f})")
        perF[qt] = {"best_single": bests, "single_auc": round(a_single, 4),
                    "lr19_auc": round(a_lr, 4), "delta": round(a_lr - a_single, 4)}
    out["per_type"] = perF

    # ---------- G. query_type as a feature ----------
    print("\n[G] query_type AS A FEATURE (caveat: hop-type may be unknown at inference)")
    for r in cal + val + test:
        r["_ismulti"] = 1.0 if r["query_type"] == "multi_hop" else 0.0
    res = {}
    for name, feats in (("LR(2)", ["dense_variance", "score_variance"]),
                        ("LR(2)+ismulti", ["dense_variance", "score_variance", "_ismulti"]),
                        ("LR(19)+ismulti", FEATURES + ["_ismulti"])):
        p = lr_pipe(); p.fit(*Xy(cal, feats))
        av = auc_of(y_bad(val), p.predict_proba(Xy(val, feats)[0])[:, 1])
        at = auc_of(y_bad(test), p.predict_proba(Xy(test, feats)[0])[:, 1])
        print(f"  {name:18s} val {av:.4f}  test {at:.4f}")
        res[name] = {"val": round(av, 4), "test": round(at, 4)}
    out["query_type_feature"] = res

    (ART / f"diagnose_classifier_{args.pool}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote artifacts/diagnose_classifier_{args.pool}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
