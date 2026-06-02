"""Build notebooks/lab.ipynb (v2.5, post-Codex corrected) with nbformat.

Teaches the cost-escalation LADDER on a MIXED workload, built up INCREMENTALLY, with the
HONEST result: adaptive routing is cost-efficient (not dominant), and the workshop's lesson
is the method, including where the loop loses. Each cell adds one piece and the attendee
reads precomputed numbers (corrected artifacts) plus a few prebuilt traces; no slow live LLM.

Result tables are rendered with pandas (clean, sortable) instead of hand-padded f-strings.
Each code cell does ONE thing (load + show a table, or render one trace); the interpretation
lives in the markdown around it, not in trailing print() blocks.

Structure: Title -> Setup -> CP1 (mixed workload + precision regime + baseline) -> CP2
(signals: the raw-dense gate + what we pruned) -> CP3 (build the ladder + the cost/quality
frontier, honest) -> STOP (gentle default vs an LLM check) -> Wrap (test headline + the
semantic judge + what we tested that did NOT win here) -> ColBERT appendix.

Reads: headline_final_v25 / judge_eval_v25 / targeted_stop_v25 / policy_comparison_val /
signal_analysis_mixed / thresholds_mixed / features_mixed_cal / demo_traces_v25.
Re-runnable: `python scripts/build_notebook.py`.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "lab.ipynb"

cells = []
def md(text): cells.append(new_markdown_cell(text.strip("\n")))
def code(src): cells.append(new_code_cell(src.strip("\n")))


# ============================================================================ title
md(r"""
# Self-Correcting Agentic Retrieval Loops

**Build an agent that reads its own retrieval, then spends only what each query needs.**

Most retrieval agents do the same thing to every query: one fixed pipeline, whether the
question is a trivial lookup or a hard multi-hop chain. In this lab you build a
**self-evaluating agent** that reads cheap in-loop signals, judges how confident its
retrieval is, and climbs a **cost-escalation ladder** only as far as it has to:

```
Tier 1  answer        confident? answer now (the cheap path)
Tier 2  ColBERT       weak single-hop lookup? a token-level precision re-retrieval
Tier 3  decompose     weak multi-hop? recover the missing hop (IRCoT)
        then ANSWER or STOP (a separate sufficiency decision)
```

The lesson is a **method**, not a recipe. Most signals and actions you try will not help on
your data; you build them, measure which earn their place, and keep only those. We run it
end-to-end on a **mixed workload** (single-hop + multi-hop + unanswerable) and report the
honest result: where adaptive routing pays, and where it does not.

Two layers, kept distinct:

- **In-loop signals = the product.** Cheap readings the agent computes *while it runs*.
- **The outer eval = ground truth.** Used only to *validate* the signals and *measure* the
  loop, never as the headline knob.

Roadmap: **CP1** the workload + baseline, **CP2** the confidence signal (and what we
pruned), **CP3** build the ladder and read the cost/quality frontier, **STOP** the
answer-vs-abstain choice, **Wrap** the honest held-out scorecard.
""")

# ============================================================================ setup
md(r"""
## Setup: run this first, confirm `Ready`

Everything below is pre-installed, pre-embedded, and warm on your VM. This cell wires
`src/` onto the path, loads your keys, and confirms Qdrant plus both collections are up.
""")
code(r"""
import sys
import os
import json
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import config, data, retrieval, signals, policy, agent
import eval as ev
import pandas as pd

pd.set_option("display.precision", 3)          # tidy floats in every table below
ARTIFACTS = REPO / "artifacts"

# Confirm Qdrant is up and both collections are populated.
client = retrieval.get_client()
main_count = client.count(config.COLLECTION, exact=True).count
assert main_count > 0, "collection empty - run scripts/setup_collections.py"

if client.collection_exists(config.COLBERT_COLLECTION):
    colbert_count = client.count(config.COLBERT_COLLECTION, exact=True).count
else:
    colbert_count = "absent"

api_keys_loaded = all(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"))

print(f"Qdrant '{config.COLLECTION}': {main_count} points")
print(f"Qdrant '{config.COLBERT_COLLECTION}': {colbert_count} points")
print(f"baseline: dense ({config.DENSE_MODEL}) + {config.HYBRID_SPARSE_VEC}, "
      f"fused with {config.FUSION_METHOD.upper()} (no cross-encoder)")
print(f"ladder tiers: answer -> ColBERT ({config.COLBERT_MODEL}) -> decompose (IRCoT) -> answer/stop")
print(f"answer context: top-{config.ANSWER_K} passages (so ranking precision is what matters)")
print(f"API keys loaded: {api_keys_loaded}")
print("\nReady" if main_count and api_keys_loaded else "\nNOT ready")
""")
md(r"""
A couple of small helpers we reuse throughout: one to read a precomputed result from
`artifacts/`, and one to pretty-print a saved agent run (its steps, top candidates, and which
signals fired).
""")
code(r"""
def load_artifact(name):
    # Read a precomputed result from artifacts/ (built by the scripts/, not computed live).
    path = ARTIFACTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing {name}: run the scripts/ that build artifacts/")
    return json.loads(path.read_text())

def load_traces(name):
    # Read a .jsonl file of saved agent traces (one JSON object per line).
    lines = (ARTIFACTS / name).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]

def frontier_table(metrics_by_policy, mrr_key, cost_key):
    # Build the cost/quality table for the four policies (used on validation and on test).
    rows = []
    for policy_name in ("always_answer", "always_colbert", "always_decompose", "ladder"):
        m = metrics_by_policy[policy_name]
        rows.append({
            "policy": policy_name.replace("_", "-"),
            "recall@3": m["recall@3"],
            "full_gold@3": m["full_gold@3"],
            "MRR": m[mrr_key],
            "LLM calls/query": m[cost_key],
        })
    return pd.DataFrame(rows)

def show_trace(trace, max_candidates=4):
    # Pretty-print one saved agent run: each step's action, top candidates, fired signals.
    print(f"Q: {trace['question']}")
    print(f"   answerable={trace['answerable']}  gold_docs={len(trace['gold_doc_ids'])}  "
          f"(the agent answers from the top-{config.ANSWER_K})")
    print("-" * 80)
    for step in trace["steps"]:
        print(f"Step {step['step']} | mode={step['mode']} -> {step['action'].upper()}  ({step['reason']})")
        if step["sub_queries"]:
            print("   sub-queries: " + " | ".join(step["sub_queries"]))
        for cand in step["candidates"][:max_candidates]:
            marker = "GOLD" if cand["is_gold"] else "    "
            print(f"   [{marker}] #{cand['rank']:<2} score={cand['score']:.3f}  {cand['title'][:58]}")
        fired = [f"{s['name']}={s['value']:.3f}" for s in step["signals"] if s["fired"]]
        print(f"   signals fired: {', '.join(fired) if fired else '(none - confident)'}"
              f"   {step['latency_ms']:.0f} ms")
    print("-" * 80)
    outcome = "STOPPED (insufficient context)" if trace["stopped"] else f"ANSWER: {trace['answer'][:72]}"
    final_candidates = trace["steps"][-1]["candidates"]
    gold_found = len({c["doc_id"] for c in final_candidates} & set(trace["gold_doc_ids"]))
    print(outcome)
    print(f"   gold in final pool: {gold_found}/{len(trace['gold_doc_ids'])}  |  "
          f"{trace['tool_calls']} tool-calls  |  {trace['total_latency_ms']:.0f} ms total")

demo_traces = {t["question_id"]: t for t in load_traces("demo_traces_v25.jsonl")}
print(f"loaded {len(demo_traces)} demo traces")
""")

# ============================================================================ CP1
md(r"""
## CP1: the mixed workload, the precision regime, and the baseline

Real traffic is **mixed**: some queries are easy single-hop lookups, some are hard multi-hop
chains, some are unanswerable. Our test slice is 120 single-hop / 60 multi-hop / 141
unanswerable, all over the same Qdrant corpus.

The **baseline** is one hybrid retrieve (dense + miniCOIL, RRF) then answer, with no loop. The
key design choice: the agent answers from a **focused top-3 context**. This is an agentic
pipeline, so the LLM reads only the top few passages; a correct passage at rank 10 is no use
to it. So ranking *precision* (recall@1/@3) is the metric that matters, and it is what gives
the corrective tiers room to work.
""")
md(r"""
### First, the two kinds of query

The whole lab turns on one distinction:

- **Single-hop**: the answer lives in one passage, and the question points straight at it.
  *"Which continent is the Atbarah River on?"* needs one retrieve, done.
- **Multi-hop**: the answer chains two or more passages, and only the **first** is reachable
  from the question as written. You learn a **bridge entity** from hop 1, and only then can you
  even phrase the query for hop 2.

A real multi-hop question from our set, *"Who is the brother of the painter of Metaphysical
Interior with Biscuits?"*, decomposes into:

1. **Hop 1**: who painted it? -> *Giorgio de Chirico* (the bridge entity)
2. **Hop 2**: who is de Chirico's brother? -> the answer

You cannot write the hop-2 query until hop 1 resolves, and the hop-2 passage shares almost no
words with the original question. So a single retrieval gets at most the first hop, no matter
how deep the top-k. That is a **recall** problem, not a ranking one, and it is exactly why
`full_gold@3` collapses below `recall@3` for multi-hop in the table below.
""")
code(r"""
headline = load_artifact("headline_final_v25.json")
by_type = headline["by_type"]

baseline_by_type = pd.DataFrame([
    {
        "query type": query_type.replace("_", "-"),
        "recall@1": by_type[query_type]["always_answer"]["recall@1"],
        "recall@3": by_type[query_type]["always_answer"]["recall@3"],
        "full_gold@3": by_type[query_type]["always_answer"]["full_gold@3"],
        "MRR": by_type[query_type]["always_answer"]["mrr_first"],
    }
    for query_type in ("single_hop", "multi_hop")
])
baseline_by_type
""")
md(r"""
**Two different failure modes.** Single-hop is nearly solved (recall@3 around 0.97). Multi-hop
is not (full_gold@3 around 0.15: both supporting passages rarely land in the top-3 together). A
precision problem and a recall problem, so one fixed pipeline cannot be right for both.

Here is the cheap path in action: a confident single-hop lookup answers at tier 1, no
escalation.
""")
code(r"""
single_hop_demo = list(demo_traces.values())[0]
show_trace(single_hop_demo)
""")
md(r"""
Now the other failure mode, the one the table predicts. Here is the **baseline** retrieve on a
multi-hop question (the first hybrid pass, before any correction). Only one of the two
supporting passages lands in the top-3. The second hop is not near this query in embedding
space, so a deeper top-k or a reranker will not surface it. That is the recall gap, and we
recover the missing hop with decomposition in CP3.
""")
code(r"""
# The same baseline retrieve, now on a MULTI-HOP question (step 1, before any correction).
multi_hop_demo = next(t for t in demo_traces.values()
                      if t["answerable"] and len(t["gold_doc_ids"]) > 1)
baseline_retrieve = multi_hop_demo["steps"][0]
gold_ids = set(multi_hop_demo["gold_doc_ids"])

print(f"Q: {multi_hop_demo['question']}\n")
print(f"baseline hybrid retrieve, top-5 ({len(gold_ids)} supporting passages needed):\n")
for cand in baseline_retrieve["candidates"][:5]:
    marker = "GOLD" if cand["is_gold"] else "    "
    print(f"   [{marker}] #{cand['rank']:<2} score={cand['score']:.3f}  {cand['title'][:55]}")

gold_in_top3 = len({c["doc_id"] for c in baseline_retrieve["candidates"][:3]} & gold_ids)
print(f"\n   gold in top-3: {gold_in_top3}/{len(gold_ids)}  "
      f"(only the first hop landed; the second supporting passage is missing)")
""")

# ============================================================================ CP2
md(r"""
## CP2: the confidence signal (and what we pruned)

The agent needs a cheap reading of "is this retrieval good enough to answer?" We test a
catalog of candidate signals and keep only what separates good from weak retrieval on
**validation**. The "good" label was fixed first: **full_gold@3** (all supporting passages in
the focused top-3). The AUCs below were precomputed; you read them (0.5 = chance, 1.0 = perfect).

The finding that matters: read the **spread of the raw dense cosine scores**
(`dense_variance`), *not* the rank-fused RRF score. RRF compresses scores into ranks and
discards the spread, so the same idea read on the fused score is much weaker. Same signal,
right substrate.
""")
code(r"""
signal_analysis = load_artifact("signal_analysis_mixed.json")
validation_auc = signal_analysis["auc_validation"]
selected_signals = signal_analysis["selection"]["weakness_signals"]

# What each signal reads, and where it earns its keep on OTHER data even when it lost here.
USEFUL_ELSEWHERE = {
    "dense_variance":       "raw-dense spread; the gate we keep here",
    "dense_gap":            "same axis as dense_variance (~0.99 correlated), keep one",
    "score_variance":       "fused spread; separates the multi-hop slice, so kept too",
    "confidence_gap":       "fused spread; redundant twin of score_variance",
    "max_score":            "classic QPP signal when scores are calibrated (single dense retriever)",
    "evidence_coverage":    "single-hop / entity lookups, where a missing entity is visible",
    "retriever_divergence": "an uncertainty signal across stacks; below the bar here",
}

signal_catalog = pd.DataFrame([
    {
        "signal": name,
        "validation AUC": validation_auc[name],
        "verdict": "kept" if name in selected_signals else "dropped",
        "useful elsewhere": USEFUL_ELSEWHERE.get(name, ""),
    }
    for name in sorted(validation_auc, key=lambda n: -(validation_auc[n] or 0))
])
signal_catalog
""")
md(r"""
The selected gate is the raw-dense spread for single-hop confidence, plus the fused spread,
which independently separated the multi-hop slice. That is the whole point of the method:
**most signals are weak here, several are useful on other stacks.** Measure on your data and
keep what clears the bar.

### Tune the confidence gate (one knob, precomputed features)

The tier-1 gate fires "weak, escalate" when the dense spread falls **below** a floor. Turn the
knob and watch precision and recall move, computed from the cached calibration features (no
Qdrant, no model calls).
""")
code(r"""
calibration = load_artifact("features_mixed_cal.json")
thresholds = load_artifact("thresholds_mixed.json")

# Label each calibration query: was retrieval WEAK (a gold passage missing from the top-3)?
retrieval_was_weak = [row["full_gold_label"] == 0 for row in calibration]
dense_spread = [row["dense_variance"] for row in calibration]

def precision_recall_at(floor):
    # Predict 'weak' when the dense spread is below `floor`, then score that prediction.
    predicted_weak = [spread < floor for spread in dense_spread]
    true_pos  = sum(pred and weak for pred, weak in zip(predicted_weak, retrieval_was_weak))
    false_pos = sum(pred and not weak for pred, weak in zip(predicted_weak, retrieval_was_weak))
    false_neg = sum((not pred) and weak for pred, weak in zip(predicted_weak, retrieval_was_weak))
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else float("nan")
    recall    = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else float("nan")
    escalation_rate = sum(predicted_weak) / len(predicted_weak)
    return precision, recall, escalation_rate

floor = thresholds["dense_variance"]      # calibrated tier-1 gate.  <-- TUNE THIS and re-run.
precision, recall, escalation_rate = precision_recall_at(floor)

print(f"dense_variance floor = {floor}")
print(f"  precision       = {precision:.3f}   (of the queries we escalate, how many were truly weak)")
print(f"  recall          = {recall:.3f}   (of the truly weak queries, how many we catch)")
print(f"  escalation rate = {escalation_rate:.2f}   (fraction of queries sent past tier 1)")
""")

# ============================================================================ CP3
md(r"""
## CP3: build the ladder, one tier at a time

Wire the gate into a policy and add the corrective tiers, each matched to a failure mode. A
weak **single-hop** lookup usually has the right passage in the pool but mis-ranked (a
*precision* problem, so ColBERT late interaction promotes it into the top-3). A weak
**multi-hop** query is missing a hop entirely (a *recall* problem, so decompose retrieves it).
We measure each on validation with a per-query counterfactual (every action on every query, no
survivorship bias).

### Tier 2: ColBERT late interaction (single-hop precision)
""")
code(r"""
policy_comparison = load_artifact("policy_comparison_val.json")
overall = policy_comparison["overall"]

tier2_colbert = pd.DataFrame([
    {"policy": "always-answer (baseline)",
     "recall@3": overall["always_answer"]["recall@3"], "MRR": overall["always_answer"]["mrr"]},
    {"policy": "always-ColBERT",
     "recall@3": overall["always_colbert"]["recall@3"], "MRR": overall["always_colbert"]["mrr"]},
])
tier2_colbert
""")
md(r"""
A wash on average. But the ladder sends ColBERT only the **weak** single-hop lookups, and on
exactly those it promotes the gold passage toward rank 1. A cross-encoder reranker ties ColBERT
here; we use ColBERT because it is a native Qdrant multivector. On your data, test both.

The trace below shows a weak lookup escalating to ColBERT:
""")
code(r"""
colbert_demo = next(
    (t for t in demo_traces.values() if any(s["action"] == "colbert" for s in t["steps"])),
    None,
)
if colbert_demo:
    show_trace(colbert_demo)
""")
md(r"""
### Tier 3: decompose (IRCoT) for multi-hop recall
""")
code(r"""
multi_hop = policy_comparison["by_type"]["multi_hop"]

tier3_decompose = pd.DataFrame([
    {"policy": "always-answer (baseline)",
     "multi-hop full_gold@3": multi_hop["always_answer"]["full_gold@3"]},
    {"policy": "always-decompose",
     "multi-hop full_gold@3": multi_hop["always_decompose"]["full_gold@3"]},
])
tier3_decompose
""")
md(r"""
Decomposition recovers the missing hop: it sharply increases how often **both** supporting
passages land in the focused context. That is the recall fix. The trace below is the same
multi-hop question that lost a hop at baseline in CP1; watch it ask the next still-missing
sub-question and pull the second passage in:
""")
code(r"""
decompose_demo = next(
    (t for t in demo_traces.values() if any(s["action"] == "decompose" for s in t["steps"])),
    None,
)
if decompose_demo:
    show_trace(decompose_demo)
""")
md(r"""
### Assemble the ladder, and read the cost/quality frontier

Put the tiers behind the gate: confident answers, weak single-hop goes to ColBERT, weak
multi-hop goes to decompose. Now compare the adaptive **ladder** against every **fixed** policy
on both axes. Cost is the mean LLM sub-query calls per query (decompose is the expensive one).
""")
code(r"""
frontier_validation = frontier_table(overall, mrr_key="mrr", cost_key="cost_llm")
frontier_validation
""")
md(r"""
Read this as a cost/quality tradeoff, not a single winner. The ladder reaches about the same
answerable quality as always-decompose at **under half** its LLM cost, by sending each query
only to the tier it needs. It **leads on MRR** (gets the right passage highest);
always-decompose **leads on full_gold@3** (completeness). So the ladder does not dominate: it is
the efficient point, most of the quality for far less cost.
""")

# ============================================================================ STOP
md(r"""
## The STOP decision: a smaller, separate lever

Stopping is a different decision from routing: not which fix to apply, but whether to answer at
all or abstain. Use the **gentle stop** by default (the generator answers, or says it lacks
enough); it keeps the most answers. For workloads where you want the highest confidence and
abstaining out of caution is fine, swap in an **LLM sufficiency check** (a fast model that reads
whether the passages actually answer the question): it catches far more unanswerables and
handles more of the full workload correctly, at the cost of occasionally refusing a question it
could have answered.
""")
code(r"""
stop_variants = load_artifact("targeted_stop_v25.json")["variants"]

stop_rows = [
    ("hybrid baseline + gentle",       "baseline_hybrid_gentle"),
    ("ladder + gentle (default)",      "ladder_gentle"),
    ("ladder + LLM sufficiency check", "ladder_autorater_all"),
]
stop_methods = pd.DataFrame([
    {
        "setup": label,
        "catches unanswerable": stop_variants[key]["abstain_unans"],
        "over-refuses answerable": stop_variants[key]["false_stop_ans"],
        "full workload handled": stop_variants[key]["selective_accuracy"],
    }
    for label, key in stop_rows
])
stop_methods
""")
md(r"""
"Full workload handled" counts a query as handled when the agent gives the right answer OR
correctly refuses an unanswerable. The gentle stop keeps answers; the LLM check nearly doubles
unanswerable abstention and wins the full workload, but over-refuses some answerables. Routing
is not stopping: a good router is not automatically a good stopper, and the ceiling on either is
retrieval completeness.

Here is a query the agent correctly refuses:
""")
code(r"""
refusal_demo = next((t for t in demo_traces.values() if t.get("stopped")), None)
if refusal_demo:
    show_trace(refusal_demo)
""")

# ============================================================================ WRAP
md(r"""
## Wrap: the honest scorecard (held-out test)

The adaptive ladder against the fixed policies on the test slice. We lead with retrieval
precision (the contamination-resistant measure of what the loop fixes) and report answer
quality with a semantic judge, not exact match. Honest caveat: this test slice partly reuses
questions from earlier rounds (disclosed in `headline_final_v25.json`), so treat it as held-out
from threshold tuning, not as a pristine never-seen set.
""")
code(r"""
overall_test = headline["overall"]

frontier_test = frontier_table(overall_test, mrr_key="mrr_first", cost_key="llm_calls")
frontier_test
""")
code(r"""
ci_vs_baseline = headline["ci_vs"]["always_answer"]
print("ladder vs baseline lift (paired bootstrap, 95% CI):\n")
for metric in ("recall@3", "full_gold@3"):
    lift = ci_vs_baseline[metric]["lift"]
    low, high = ci_vs_baseline[metric]["ci95"]
    verdict = "clears 0" if low > 0 else "crosses 0"
    print(f"  {metric:12s} {lift:+.4f}   CI95 [{low:+.4f}, {high:+.4f}]   {verdict}")
""")
md(r"""
Answer quality uses a **gpt-5.5 semantic judge** (it credits correct-but-paraphrased answers),
not exact match.
""")
code(r"""
judge = load_artifact("judge_eval_v25.json")
by_policy = judge["by_policy"]

answer_quality = pd.DataFrame([
    {
        "policy": policy_name.replace("_", "-"),
        "overall": by_policy[policy_name]["overall"]["judge"],
        "single-hop": by_policy[policy_name]["single_hop"]["judge"],
        "multi-hop": by_policy[policy_name]["multi_hop"]["judge"],
    }
    for policy_name in ("always_answer", "ladder", "always_decompose")
])
answer_quality
""")
code(r"""
judge_ci = judge["ci_ladder"]
print(f"ladder vs baseline:  {judge_ci['vs_baseline']['lift']:+.3f}   "
      f"CI95 {judge_ci['vs_baseline']['ci95']}  (crosses 0, so the lift is marginal)")
print(f"ladder vs decompose: {judge_ci['vs_decompose']['lift']:+.3f}   (decompose answers best)")
""")
md(r"""
### What we learned (and what we honestly did not)

- **Adaptive routing is cost-efficient, not dominant.** The ladder reaches near-decompose
  answerable quality at about 40% of always-decompose's LLM cost and beats the no-correction
  baseline on retrieval (CIs clear of zero). It leads MRR (rank); always-decompose leads
  full_gold@3 (completeness). It does not win every metric, it wins the cost/quality tradeoff.
- **The gains are per-slice.** Decomposition lifts multi-hop full_gold and multi-hop answer
  quality by tens of points; the overall lift is small because the easy single-hop majority is
  already strong. That dilution *is* the cost story: easy queries stay cheap.
- **The right substrate makes a weak signal strong.** Reading spread on raw dense cosines, not
  the rank-fused score, is what made the confidence gate work.
- **Routing is not stopping.** A real sufficiency check wins the full workload (handles about
  0.63 vs the baseline's 0.53) but trades away some answers. Pick the stop for your tolerance.
- **Honesty on decompose.** IRCoT's sub-queries are written by an LLM that may know the bridge
  entities, so its lift is an upper bound that shrinks on unseen corpora. Recalibrate on yours.

**What we tested that did NOT win here (but wins elsewhere)**, the heart of "validate, don't
assume": the `evidence_coverage` and `retriever_divergence` signals (useful on single-hop /
entity-lookup / lexical-mismatch data); `max_score` (a calibrated single-retriever QPP signal);
a cross-encoder reranker (tied ColBERT here); and the recall@10 framing (it hid all the
headroom, because single-hop is about 98% solved at top-10). None is useless: each would win on
a different workload. The deliverable is the **method**: build cheap signals, validate which
predict and fix bad retrieval *on your data*, route to the cheapest sufficient action, and
measure on cost and quality. Take home: this notebook and the reusable `src/` modules.
""")

# ============================================================================ appendix (ColBERT)
md(r"""
## Appendix: Qdrant native multivector (ColBERT late interaction)

Tier 2 is ColBERT late interaction, stored and scored natively by Qdrant (MaxSim over
token-level multivectors). This cell runs one live query against the `musique_colbert`
collection so you can see the mechanism: prefetch a dense pool, then rescore with MaxSim.
""")
code(r"""
from qdrant_client import models

question = "Who developed The Genius of Victory?"

if client.collection_exists(config.COLBERT_COLLECTION):
    dense_model = retrieval.get_models()["dense"]
    colbert_model = retrieval.get_colbert_model()
    dense_query = next(iter(dense_model.query_embed(question))).tolist()
    colbert_query = [token_vec.tolist() for token_vec in next(iter(colbert_model.query_embed(question)))]

    results = client.query_points(
        config.COLBERT_COLLECTION,
        prefetch=[models.Prefetch(query=dense_query, using=config.DENSE_VEC, limit=50)],
        query=colbert_query,
        using=config.COLBERT_VEC,
        limit=5,
        with_payload=True,
    ).points

    print(f"ColBERT MaxSim top-5 for: {question}\n")
    for rank, point in enumerate(results, start=1):
        print(f"  #{rank}  MaxSim {point.score:.2f}  {point.payload.get('title', '')[:60]}")
else:
    print(f"{config.COLBERT_COLLECTION} absent. Run scripts/setup_colbert.py to build it.")
""")

# ============================================================================ write
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
