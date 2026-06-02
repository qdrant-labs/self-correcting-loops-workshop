"""In-loop retrieval-quality signals (the product).

Computed from a RetrievalResult. Two families of weakness signal:
  - FUSION-score signals read the active fused (RRF) scores;
  - RAW-DENSE signals read the pre-fusion dense cosine scores, where the spread the
    signal needs still has dynamic range. RRF compresses scores into ranks and
    flattens that range, so reading spread on the raw dense scores is the headline
    finding here: it lifts the confidence AUC from ~0.62 (fused) to ~0.78 (raw dense).

Candidate signals across the axes; validation selection (calibrate_mixed.py) keeps
the non-redundant set that actually separates good from weak retrieval on the data at
hand. On the mixed MuSiQue workload the kept gate is {dense_variance, score_variance};
the rest are cataloged as "weak here, useful elsewhere" (see the tutorial):

  dense_variance       spread (pstdev) of top-K RAW DENSE cosines       [spread, raw]    <- gate
  dense_gap            rank-1 minus rank-K RAW DENSE cosine             [spread, raw]    (~dense_variance)
  score_variance       spread (pstdev) of top-K fused scores            [spread, fused]  <- gate
  confidence_gap       rank-1 minus rank-K fused score                  [spread, fused]
  max_score            top-1 fused score                                [height]
  evidence_coverage    fraction of question entities present in top-K   [coverage] (cheap; no LLM)
  retriever_divergence dense vs BM25 top-K disagreement                 [agreement]

Each returns a SignalReading(name, value, threshold, fired). `fired` means "retrieval
looks weak": the height/spread/coverage signals fire BELOW their floor; divergence
fires ABOVE its ceiling (disagreement). Only the SELECTED weakness signals gate
`healthy`; the others are still computed (so a trace can show them) but do not decide.

evidence_coverage doubles as a cheap STOP gate ("are the question's named entities
even present in the retrieved evidence?"). It detects weakness, not unanswerability -
that gap is what motivates the LLM sufficiency autorater (the STOP upgrade).

Divergence reads the raw (pre-fusion) dense vs BM25 rankings. Thresholds are
calibrated per dataset on the calibration split and loaded from
artifacts/thresholds_mixed.json; the defaults below are uncalibrated, NON-portable
placeholders.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import string
from dataclasses import dataclass

import config

DEFAULT_DETECTOR = config.DIVERGENCE_SPARSE_VEC  # "bm25"

# Uncalibrated placeholders. Calibration overwrites these and sets _calibrated=True;
# the headline/test stages refuse to run on placeholders.
DEFAULT_THRESHOLDS = {
    "max_score": 0.03,            # top-1 fusion floor (fires below)
    "score_variance": 0.005,      # pstdev floor of top-K fusion scores (fires below)
    "confidence_gap": 0.01,       # rank1 - rankK fusion floor (fires below)
    "evidence_coverage": 0.5,     # entity-coverage floor (fires below)
    "retriever_divergence": 0.60, # overlap-divergence ceiling (fires above)
    # v2.5: spread on the RAW DENSE cosine scores (NOT the rank-compressed fusion).
    # This is the strong tier-1 confidence signal - a peaked dense ranking (large
    # rank1-rankK gap) means a confident lookup. Validation AUC ~0.75 (vs the v2
    # fusion signals' ~0.62 ceiling), because RRF's rank-compression flattens height/spread.
    "dense_gap": 0.05,            # rank1 - rankK RAW DENSE cosine floor (fires below)
    "dense_variance": 0.02,       # pstdev floor of top-K RAW DENSE cosine (fires below)
    "_calibrated": False,
}

# Selected by validation; the live gate honors these (calibrated thresholds carry
# `_weakness_signals`). Falls back to the v2.5 selected gate if the loaded thresholds
# omit the selection.
DEFAULT_WEAKNESS = ("dense_variance", "score_variance")


@dataclass
class SignalReading:
    name: str
    value: float
    threshold: float
    fired: bool
    detail: str


@dataclass
class SignalReport:
    readings: list[SignalReading]
    healthy: bool                  # none of the SELECTED weakness signals fired
    fired: dict[str, bool]
    weakness: tuple = ()           # the signals that gate `healthy` (post-selection)

    def get(self, name: str) -> SignalReading:
        return next(r for r in self.readings if r.name == name)

    def value(self, name: str) -> float:
        return self.get(name).value


def load_thresholds(path=None) -> dict:
    p = path or (config.ARTIFACTS_DIR / "thresholds_mixed.json")
    th = dict(DEFAULT_THRESHOLDS)
    if p.exists():
        th.update(json.loads(p.read_text()))
    return th


# --- evidence coverage (cheap entity match; no LLM) --------------------------
_QUESTION_STOP = {
    "what", "who", "whom", "whose", "where", "when", "which", "why", "how",
    "is", "was", "are", "were", "did", "do", "does", "the", "a", "an", "name",
    "in", "of", "on", "at", "to", "for", "by", "as", "that", "this",
}
_NAME_CONNECTORS = {"of", "the", "and", "de", "von", "van", "del", "la", "el", "da", "di", "&"}


def question_entities(question: str) -> set[str]:
    """Teaching-simple entity extractor: maximal runs of Capitalized tokens (joined by
    lowercase name-connectors like 'of'/'the'), plus 4-digit years. Drops the
    sentence-initial question word. The production upgrade is spaCy NER - mentioned
    in the docs, not a live dependency."""
    toks = (question or "").split()
    ents, cur = [], []
    for i, tok in enumerate(toks):
        w = tok.strip(string.punctuation)
        if not w:
            if cur:
                ents.append(" ".join(cur)); cur = []
            continue
        is_cap = w[0].isupper()
        if is_cap and not (i == 0 and w.lower() in _QUESTION_STOP):
            cur.append(w)
        elif cur and w.lower() in _NAME_CONNECTORS and i + 1 < len(toks) \
                and toks[i + 1].strip(string.punctuation)[:1].isupper():
            cur.append(w)  # lowercase connector inside a name ("University of Texas")
        elif cur:
            ents.append(" ".join(cur)); cur = []
    if cur:
        ents.append(" ".join(cur))
    out = {e.lower() for e in ents if len(e) >= 2}
    out |= set(re.findall(r"\b\d{4}\b", question or ""))
    return out


def evidence_coverage(result, k: int = config.TOP_K) -> float:
    """Fraction of the question's named entities that appear (substring, case-insensitive)
    in the retrieved top-K passages. Low coverage => the retrieved evidence does not even
    mention what the user named => weak retrieval. Returns 1.0 if no entities (nothing to
    miss)."""
    ents = question_entities(getattr(result, "query", ""))
    if not ents:
        return 1.0
    blob = " ".join(f"{c.title} {c.text}" for c in result.candidates[:k]).lower()
    return sum(1 for e in ents if e in blob) / len(ents)


# --- divergence measures (computed on raw, pre-fusion rankings) --------------
def _overlap_divergence(dense_ids, sparse_ids, k) -> float:
    """Teaching simplification: 1 - |dense_topK n sparse_topK| / K. 0 = agree, 1 = disjoint."""
    a, b = dense_ids[:k], sparse_ids[:k]
    if not a or not b:
        return 1.0 if (a or b) else 0.0
    return 1.0 - len(set(a) & set(b)) / max(len(a), len(b))


def _rbo(a, b, p: float = 0.9) -> float:
    """Rank-biased overlap similarity (top-weighted; handles disjoint / unequal lists).
    The 'ordered' divergence option vs the set-overlap simplification."""
    depth = min(len(a), len(b))
    if depth == 0:
        return 0.0
    cum = sum((p ** (d - 1)) * (len(set(a[:d]) & set(b[:d])) / d) for d in range(1, depth + 1))
    return (1 - p) * cum / (1 - p ** depth)


def divergence(raw, detector: str = DEFAULT_DETECTOR, k: int = config.TOP_K, measure: str = "overlap") -> float:
    """Disagreement between the dense ranking and a sparse ranking. measure: 'overlap'
    (taught) or 'rbo' (ordered). NaN if the raw rankings are absent."""
    if not raw or "dense" not in raw or detector not in raw:
        return float("nan")
    dense_ids = [i for i, _ in raw["dense"]][:k]
    sparse_ids = [i for i, _ in raw[detector]][:k]
    if measure == "rbo":
        return 1.0 - _rbo(dense_ids, sparse_ids)
    return _overlap_divergence(dense_ids, sparse_ids, k)


# --- raw-dense spread (the v2.5 tier-1 confidence substrate) ------------------
def dense_spread(result, k: int = config.TOP_K) -> tuple[float, float]:
    """(gap, variance) of the RAW DENSE cosine scores over the top-k pre-fusion ranking.
    Read on raw dense - not the RRF-fused score - because RRF compresses scores to ranks
    and flattens the dynamic range these signals need. A peaked dense ranking (large
    rank1-rankK gap, high variance) means a confident lookup; a flat one means the dense
    retriever cannot tell its top candidates apart -> weak. Returns (0.0, 0.0) if the raw
    dense ranking is absent."""
    raw = (result.raw or {}).get("dense") or []
    ds = [s for _, s in raw][:k]
    if len(ds) < 2:
        return 0.0, 0.0
    return ds[0] - ds[-1], statistics.pstdev(ds)


# --- the two entry points ----------------------------------------------------
def signal_values(result, detector: str = DEFAULT_DETECTOR, k: int = config.TOP_K) -> dict:
    """Raw signal VALUES (no thresholds), for the offline feature matrix / AUC /
    calibration. Fusion-score signals read the ACTIVE scores; dense_* read the raw dense
    cosines (the v2.5 strong substrate)."""
    scores = list(result.scores)
    d_gap, d_var = dense_spread(result, k)
    return {
        "max_score": scores[0] if scores else 0.0,
        "score_variance": statistics.pstdev(scores) if len(scores) >= 2 else 0.0,
        "confidence_gap": (scores[0] - scores[-1]) if len(scores) >= 2 else 0.0,
        "evidence_coverage": evidence_coverage(result, k),
        "dense_gap": d_gap,
        "dense_variance": d_var,
        "divergence_overlap": divergence(result.raw, detector, k, "overlap"),
        "divergence_rbo": divergence(result.raw, detector, k, "rbo"),
    }


def read_signals(result, thresholds: dict | None = None, detector: str = DEFAULT_DETECTOR) -> SignalReport:
    """The live in-loop read: values + fired flags + a `healthy` verdict gated by the
    SELECTED weakness signals."""
    th = thresholds or load_thresholds()
    scores = list(result.scores)
    n = len(scores)

    ms = scores[0] if scores else 0.0
    var = statistics.pstdev(scores) if n >= 2 else 0.0
    gap = (scores[0] - scores[-1]) if n >= 2 else 0.0
    cov = evidence_coverage(result)
    d_gap, d_var = dense_spread(result)
    div = divergence(result.raw, detector)
    div_rbo = divergence(result.raw, detector, measure="rbo")

    readings = [
        SignalReading("max_score", ms, th["max_score"], ms < th["max_score"],
                      f"top-1 fusion {ms:.4f} vs floor {th['max_score']:.4f}"),
        SignalReading("score_variance", var, th["score_variance"], var < th["score_variance"],
                      f"pstdev(top-{n}) {var:.4f} vs floor {th['score_variance']:.4f}"),
        SignalReading("confidence_gap", gap, th["confidence_gap"], gap < th["confidence_gap"],
                      f"rank1-rankK {gap:.4f} vs floor {th['confidence_gap']:.4f}"),
        SignalReading("evidence_coverage", cov, th["evidence_coverage"], cov < th["evidence_coverage"],
                      f"entity coverage {cov:.3f} vs floor {th['evidence_coverage']:.3f}"),
        SignalReading("dense_gap", d_gap, th["dense_gap"], d_gap < th["dense_gap"],
                      f"raw-dense rank1-rankK {d_gap:.4f} vs floor {th['dense_gap']:.4f}"),
        SignalReading("dense_variance", d_var, th["dense_variance"], d_var < th["dense_variance"],
                      f"raw-dense pstdev {d_var:.4f} vs floor {th['dense_variance']:.4f}"),
        SignalReading(
            "retriever_divergence", div, th["retriever_divergence"],
            (not math.isnan(div)) and div > th["retriever_divergence"],
            f"overlap-div {div:.3f} (rbo-div {div_rbo:.3f}) vs ceil {th['retriever_divergence']:.3f}; detector={detector}",
        ),
    ]
    fired = {r.name: r.fired for r in readings}
    weakness = tuple(th.get("_weakness_signals") or DEFAULT_WEAKNESS)
    healthy = not any(fired.get(w) for w in weakness)
    return SignalReport(readings=readings, healthy=healthy, fired=fired, weakness=weakness)
