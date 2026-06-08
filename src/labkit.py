"""Rendering helpers for notebooks/lab.ipynb.

PRESENTATION ONLY. Everything the workshop actually teaches (the Qdrant queries, the signals,
the gate, the decompose loop, the assembled agent) is written and shown live in the notebook;
this module just keeps the printing and plotting out of the teaching cells. Data is read from
artifacts/ (built by the scripts/) only for the aggregate eval scorecards.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

ARTIFACTS = config.ARTIFACTS_DIR
GOOD, WEAK, ACCENT = "#1f9d55", "#d64545", "#D6336C"


def load_artifact(name):
    """Read a precomputed result from artifacts/ (built by the scripts/, not computed live)."""
    path = ARTIFACTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing {name}: run the scripts/ that build artifacts/")
    return json.loads(path.read_text())


def _id_title_text(hit):
    """Read (doc_id, title, text) from either a raw Qdrant ScoredPoint or a Candidate."""
    if hasattr(hit, "payload"):                      # raw Qdrant point
        return hit.id, hit.payload.get("title", ""), hit.payload.get("text", "")
    return hit.doc_id, hit.title, hit.text           # retrieval.Candidate


def show_hits(hits, gold_ids, k=3, snippet=95):
    """Print the top-k retrieved passages with a content snippet, marking the gold ones.

    Accepts raw Qdrant points (id/payload) or Candidates, so it renders whatever the live
    query returned.
    """
    gold = set(gold_ids)
    for rank, hit in enumerate(hits[:k], start=1):
        doc_id, title, text = _id_title_text(hit)
        marker = "GOLD" if doc_id in gold else "    "
        print(f"  [{marker}] #{rank}  {title}")
        print(f"            {' '.join((text or '').split())[:snippet]}...")


def show_run(question, route, answer, hits, gold_ids):
    """Summarize one assembled-loop run: the path it took, the answer, and gold coverage."""
    gold = set(gold_ids)
    print(f"Q: {question}")
    print(f"  route:  {route}")
    print(f"  answer: {answer.strip()[:72]}")
    if gold:
        found = len({_id_title_text(h)[0] for h in hits[:3]} & gold)
        print(f"  gold in the answer context: {found}/{len(gold)}")
    print()


def frontier_table(metrics_by_policy, mrr_key, cost_key):
    """Build the cost/quality table for the four policies (used on validation and on test)."""
    rows = []
    for policy_name in ("always_answer", "always_colbert", "always_decompose", "ladder"):
        m = metrics_by_policy[policy_name]
        row = {
            "policy": policy_name.replace("_", "-"),
            "recall@3": m["recall@3"],
            "full_gold@3": m["full_gold@3"],
            "MRR": m[mrr_key],
            "LLM calls/query": m[cost_key],
        }
        if "avg_latency_s" in m:
            row["avg routing latency (s)"] = m["avg_latency_s"]
        elif "avg_latency_ms" in m:
            row["avg routing latency (s)"] = round(m["avg_latency_ms"] / 1000, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_signal_separation(features, signal_auc, kept, column):
    """Boxplot each candidate signal's value on good vs weak retrievals; separation is its AUC.

    `signal_auc` and `column` (signal -> feature key) are computed in the notebook, so the plot
    shows exactly the benchmark the reader just ran.
    """
    good = [r for r in features if r["full_gold_label"] == 1]
    weak = [r for r in features if r["full_gold_label"] == 0]
    order = sorted(column, key=lambda s: -signal_auc[s])

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, name in zip(axes.flat, order):
        col = column[name]
        boxes = ax.boxplot([[r[col] for r in good], [r[col] for r in weak]],
                           tick_labels=["good", "weak"], widths=0.6,
                           patch_artist=True, showfliers=False)
        boxes["boxes"][0].set(facecolor=GOOD, alpha=0.55)
        boxes["boxes"][1].set(facecolor=WEAK, alpha=0.55)
        verdict = "KEPT" if name in kept else "dropped"
        ax.set_title(f"{name}\nAUC {signal_auc[name]:.2f} ({verdict})", fontsize=9.5,
                     color=ACCENT if name in kept else "black")
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    fig.suptitle("Each signal on good vs weak retrievals (validation): separation = predictive power",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


_GATE_SIGNAL_DESC = {
    "dense_variance": "spread of the raw dense scores",
    "score_variance": "spread of the fused RRF scores",
}


def plot_gate(features, floor, signal="dense_variance", desc=None):
    """The chosen signal's good-vs-weak distributions, with the floor and what it buys
    (recall = weak caught; escalation = share of all queries sent past tier 1).
    Defaults to dense_variance; pass signal="score_variance" + SV_FLOOR for the other gate."""
    desc = desc or _GATE_SIGNAL_DESC.get(signal, signal)
    good = [r[signal] for r in features if r["full_gold_label"] == 1]
    weak = [r[signal] for r in features if r["full_gold_label"] == 0]
    values = np.array([r[signal] for r in features])
    is_weak = np.array([r["full_gold_label"] == 0 for r in features])

    if not values.min() <= floor <= values.max():   # mismatched signal/floor -> fail loud, not a 100% plot
        raise ValueError(
            f"floor {floor:.3f} is outside the {signal} range [{values.min():.3f}, {values.max():.3f}]; "
            f"pass the matching signal, e.g. plot_gate(..., SV_FLOOR, signal='score_variance')."
        )

    below = values < floor
    recall = np.sum(below & is_weak) / max(np.sum(is_weak), 1)   # of weak retrievals, how many fall below the floor
    escalation = below.mean()                                    # of all queries, how many fall below the floor

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, values.max(), 26)
    ax.hist(good, bins=bins, alpha=0.6, color=GOOD, label="good (full_gold present)")
    ax.hist(weak, bins=bins, alpha=0.6, color=WEAK, label="weak (full_gold missing)")
    ax.axvspan(0, floor, color="black", alpha=0.06, label="escalate (below the floor)")
    ax.axvline(floor, color="black", ls="--", lw=1.6, label=f"gate floor = {floor:.3f}")

    ax.set_title(
        "What the gate sees: low spread predicts weak retrieval\n"
        f"{signal} floor {floor:.3f}  ->  catches {recall:.0%} of weak retrievals, escalates {escalation:.0%} of all queries",
        fontsize=11,
    )
    ax.set_xlabel(f"{signal}  ({desc})")
    ax.set_ylabel("queries")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    plt.show()
