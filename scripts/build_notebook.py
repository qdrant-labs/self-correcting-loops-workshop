"""Build notebooks/lab.ipynb (v2.5, build-from-primitives, fully live).

A BUILD-IT-FROM-PRIMITIVES lab: the notebook defines the agent itself - embedding, the Qdrant
queries (hybrid fusion, native multivector), every signal, the gate, the AUC benchmark, the
decompose loop, the assembled router, the stop - as readable functions IN the cells, built from
external primitives (Qdrant, FastEmbed, an LLM). It imports NOTHING from our own retrieval/signals/
agent modules; `src/` is that same logic packaged for when the reader wants to reuse it, and
`src/labkit.py` is rendering only. The signal benchmark is computed LIVE over the calibration split;
only the four slow eval scorecards (every policy over the whole workload with the LLM, ~45 min,
must be deterministic) stay precomputed in artifacts/ and are clearly labeled.

Each inline function mirrors src/ exactly (verified: the hybrid top-1, every signal value, the AUCs,
and the Youden floors all reproduce to the digit), so the agent you watch get built is the real one.

Structure: Title -> Setup -> the LLM call -> CP1 (the hybrid query + baseline) -> CP2 (define every
signal, benchmark them live, build the gate) -> CP3 (ColBERT + decompose, run live) -> the assembled
loop (solve(), live) -> STOP (gentle vs LLM check, live refusal) -> Wrap (held-out scorecard + judge
+ how to adapt this to your workflow).
Re-runnable: `python scripts/build_notebook.py`. Honest result: adaptive routing is cost-efficient,
not dominant.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "lab.ipynb"

cells = []
def md(text): cells.append(new_markdown_cell(text.strip("\n")))
def code(src): cells.append(new_code_cell(src.strip("\n")))


def ircot_diagram_md():
    """Render the IRCoT loop as a flow diagram and return it as a self-contained markdown image
    (PNG data URI, so it travels inside the .ipynb and regenerates whenever this script runs).
    This is an explanatory figure, not agent code, so it lives in markdown, not a live cell."""
    RETR, RETR_F = "#1c7ed6", "#e7f0fb"   # retrieval steps
    LLM, LLM_F = "#D6336C", "#fce4ec"     # the LLM decompose step + the loop
    ANS, ANS_F = "#1f9d55", "#e6f4ea"     # the terminal answer/stop
    GREY, GREY_F, INK = "#5f6368", "#f1f3f4", "#202124"

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(cx, cy, w, h, text, edge, face, fs=10.5, bold=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=1.8,
            edgecolor=edge, facecolor=face, mutation_aspect=1))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=INK, rad=0.0, lw=2.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
            lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))

    yT = 2.35
    box(1.05, yT, 1.7, 0.95, "Question", GREY, GREY_F, bold=True)
    box(3.85, yT, 2.3, 0.95, "Retrieve\n(hybrid: dense\n+ sparse, RRF)", RETR, RETR_F)
    box(6.75, yT, 2.3, 0.95, "Read the\nevidence so far", RETR, RETR_F)
    box(9.95, yT, 2.6, 0.95, "LLM decomposer:\nnext missing\nsub-question?", LLM, LLM_F)
    box(9.95, 0.55, 3.4, 0.9, "Union the evidence  ->  Answer / STOP", ANS, ANS_F, fs=10)

    arrow(1.9, yT, 2.68, yT); arrow(5.0, yT, 5.6, yT); arrow(7.9, yT, 8.62, yT)
    arrow(9.95, yT + 0.49, 3.85, yT + 0.49, color=LLM, rad=0.32, lw=2.2)   # loop-back, bowing above the row
    ax.text(6.9, 3.83, "still missing a hop: ask it, retrieve again", ha="center", va="center",
            fontsize=9.5, color=LLM, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))
    arrow(9.95, yT - 0.49, 9.95, 1.02, color=ANS, lw=2.2)
    ax.text(10.18, 1.55, "ENOUGH", ha="left", va="center", fontsize=9.5, color=ANS, fontweight="bold")
    ax.text(0.2, 4.45, "IRCoT: retrieve, read, ask the next hop, repeat until the evidence is enough",
            ha="left", va="center", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return (f"![IRCoT: the iterative retrieve-and-reason loop]({uri})\n\n"
            "*The IRCoT loop. CP1 below runs a single pass of it (one follow-up hop); CP3 closes "
            "the loop into `decompose()`.*")


# ============================================================================ title
md(r"""
# Self-Correcting Agentic Retrieval Loops

**Build an agent that reads its own retrieval, then spends only what each query needs.**

Most retrieval agents do the same thing to every query: one fixed pipeline, whether the question
is a trivial lookup or a hard multi-hop chain. Here you build, from Qdrant primitives, a
**self-evaluating agent** that reads cheap in-loop signals and climbs a **cost-escalation ladder**
only as far as it has to:

```
Tier 1  answer        confident? answer now (the cheap path)
Tier 2  ColBERT       weak single-hop lookup? a token-level precision re-retrieval
Tier 3  decompose     weak multi-hop? recover the missing hop (IRCoT)
        then ANSWER or STOP (a separate sufficiency decision)
```

This notebook teaches the **method** *and* how to reproduce it. We import nothing from our own
`retrieval` / `signals` / `agent` modules: you build each of them here from external primitives
(Qdrant, FastEmbed, an LLM), so every retrieval, signal, and routing decision is real code you can
read and run. Nothing core is hidden. `src/` is that same logic packaged for when you want to reuse
it; `src/labkit.py` is rendering only. We run it end-to-end on a **mixed workload** (single-hop +
multi-hop + unanswerable) and report the honest result: where adaptive routing pays, and where it
does not.

Roadmap: **CP1** the hybrid query + baseline, **CP2** define and benchmark the confidence signals,
**the gate** turn them into a weak/strong decision, **CP3** the ColBERT and decompose tiers run live,
**the assembled loop** run end to end, **STOP** the answer-vs-abstain choice, **Wrap** the honest
scorecard and how to adapt this to your workflow.
""")

# ============================================================================ setup
md(r"""
## Setup: run this first, confirm `Ready`

Everything is pre-installed and pre-embedded on your VM. This cell imports the external primitives
(the Qdrant client, the FastEmbed models, the LLM), defines the inline constants the agent uses,
loads the question set, and **warms the embedding models** (FastEmbed fetches them from Hugging Face
the first time; on the prebuilt VM they are already cached, so this is just a load). The only thing
we pull from our own `src/` is `labkit`, which does rendering only (printing hits, the two plots).
""")
code(r"""
import sys
import os
import json
import re
import string
import time
import statistics
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import litellm
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from sklearn.metrics import roc_auc_score

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))    # ONLY so labkit (rendering) resolves; nothing core is imported from src
load_dotenv(REPO / ".env")
os.environ.setdefault("LITELLM_LOG", "ERROR")
pd.set_option("display.precision", 3)

from labkit import (load_artifact, frontier_table, show_hits, show_run,
                    plot_signal_separation, plot_gate)

# --- inline constants (the collection schema, model ids, retrieval + answer sizes) ---
COLLECTION = "musique"
COLBERT_COLLECTION = "musique_colbert"             # dense + colbert multivector (Tier 2 showcase)
DENSE_MODEL = "BAAI/bge-base-en-v1.5"              # 768-d dense, cosine
MINICOIL_MODEL = "Qdrant/minicoil-v1"             # word-sense-aware sparse (the hybrid baseline's sparse)
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
DENSE_VEC, MINICOIL_VEC, COLBERT_VEC = "dense", "minicoil", "colbert"
RETRIEVE_N = 50          # per-retriever prefetch depth before fusion
TOP_K = 10               # signal / pool window
ANSWER_K = 3             # focused passages the LLM reads to answer (the precision regime)
AGENT_MODEL = "anthropic/claude-sonnet-4-6"        # decompose + answer
FAST_MODEL = "anthropic/claude-haiku-4-5"          # the fast sufficiency autorater (STOP)

client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"), timeout=120)

# The query-side embedders. The corpus is already indexed in Qdrant; only these query models load here.
dense_model = TextEmbedding(DENSE_MODEL)
minicoil_model = SparseTextEmbedding(MINICOIL_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)
for warm in (dense_model, minicoil_model):
    next(iter(warm.query_embed("warm up")))
next(iter(colbert_model.query_embed("warm up")))

# The question set (mixed workload), loaded straight from the dataset file.
by_id = {q["id"]: q for q in (json.loads(line) for line in (REPO / "data/questions_mixed.jsonl").open())}

main_count = client.count(COLLECTION, exact=True).count
assert main_count > 0, "collection empty - run scripts/setup_collections.py"
colbert_count = (client.count(COLBERT_COLLECTION, exact=True).count
                 if client.collection_exists(COLBERT_COLLECTION) else "absent")
api_keys_loaded = all(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"))

print(f"Qdrant '{COLLECTION}': {main_count} points  |  '{COLBERT_COLLECTION}': {colbert_count} points")
print(f"dense ({DENSE_MODEL}) + sparse (minicoil), fused with RRF")
print(f"loaded {len(by_id)} questions; embedding models warm; answer context = top-{ANSWER_K}")
print(f"API keys loaded: {api_keys_loaded}")
print("\nReady" if main_count and api_keys_loaded else "\nNOT ready")
""")

# ============================================================================ the LLM call
md(r"""
## The LLM call

The agent uses an LLM for exactly three focused jobs, all routed through one helper: writing the
next decompose sub-question (CP1, CP3), generating the final grounded answer (the assembled loop),
and the sufficiency autorater (STOP). Here is that single helper. It is deterministic at
`temperature=0` and retries a few times on transient API errors so one network blip does not kill a
live run.
""")
code(r"""
def ask_llm(system, user, max_tokens=256, model=AGENT_MODEL, temperature=0.0):
    # one LLM turn via LiteLLM. Returns the message text. Deterministic at temperature 0.
    litellm.suppress_debug_info = True
    for attempt in range(4):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature, timeout=45,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
""")

# ============================================================================ CP1
md(r"""
## CP1: the hybrid query, the precision regime, and the baseline

Real traffic is **mixed**: easy single-hop lookups, hard multi-hop chains, and unanswerable
questions. The **baseline** is one hybrid retrieve then answer, no loop. The key design choice:
the agent answers from a **focused top-3 context**, so ranking *precision* (recall@1/@3) is what
matters, and it is what gives the corrective tiers room to work.

> **On your data:** set this answer-context size deliberately. At a generous top-10 an easy lookup
> is already solved and there is nothing to fix; at top-3 there is real headroom for the tiers.

First, the retrieval primitive. `embed` runs the three query-side encoders; `hybrid_search` is the
actual Qdrant **hybrid** query: dense (bge) and sparse (miniCOIL) prefetched in parallel, then fused
server-side with Reciprocal Rank Fusion. We also define the lightweight `Passage` the rest of the
loop reads. We will reuse all of this everywhere.
""")
code(r"""
def embed(text):
    # the two query-side embeddings used everywhere: dense (bge) + miniCOIL sparse. query_embed
    # applies the bge query instruction and the sparse query-side weighting; Qdrant applies IDF.
    dense = next(iter(dense_model.query_embed(text))).tolist()
    minicoil = next(iter(minicoil_model.query_embed(text)))
    return dense, minicoil

def hybrid_search(question, limit=TOP_K, enc=None):
    # Qdrant hybrid retrieval: dense + miniCOIL prefetched (top-50 each), fused server-side with RRF.
    dense, minicoil = enc or embed(question)
    return client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using=DENSE_VEC, limit=RETRIEVE_N),
            models.Prefetch(
                query=models.SparseVector(indices=minicoil.indices.tolist(),
                                          values=minicoil.values.tolist()),
                using=MINICOIL_VEC, limit=RETRIEVE_N),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit, with_payload=True,
    ).points

@dataclass
class Passage:
    # the lightweight passage object the answerer and the decompose pooler read.
    doc_id: str
    title: str
    text: str
    score: float

def to_passages(points):
    return [Passage(p.id, p.payload["title"], p.payload["text"], p.score) for p in points]
""")
md(r"""
### Two kinds of query, and why you can't tell them apart up front

Both of these are perfectly normal things a user asks, and on the surface they look the same:

- **Simple (single-hop):** the answer sits in one passage, so a single retrieve can answer it
  right away. *"Which continent is the Atbarah River on?"*
- **Complex (multi-hop):** the answer is spread across passages, and the later ones are not
  reachable from the question as written. The agent has to retrieve, read what came back, and
  retrieve again. *"What sea washes the shores of the birthplace of Jim Wilson?"* needs the
  birthplace before it can ask about the sea.

Nothing on the surface of a question reliably says "this one needs a second hop," so you cannot
sort them up front. That is exactly why the agent reads a cheap signal off its own retrieval (CP2)
and escalates only when the retrieval looks weak, instead of trusting the question.

A single-hop lookup first: the supporting passage is right there, so the agent answers from the
top-3.
""")
code(r"""
single = by_id["2hop__101521_42157__h0"]
print(f"Q: {single['question']}\n")
show_hits(hybrid_search(single["question"]), single["gold_doc_ids"])
""")
md(r"""
Now a multi-hop question. The follow-up hop is written by the agent's **decomposer**, an LLM step we
define right here (CP3 builds it into the full tier). Given the main question and the evidence so far,
it asks the next still-missing sub-question, or says `ENOUGH`. That iterative retrieve-read-ask cycle
is **IRCoT**:
""")
md(ircot_diagram_md())
code(r"""
IRCOT_SYSTEM = (
    "You are running an iterative retrieve-and-reason loop to answer a multi-hop "
    "question. Given the main question, the evidence retrieved so far, and the "
    "sub-questions already asked, output the NEXT single sub-question whose answer is "
    "still MISSING and is needed to answer the main question. Make it self-contained: "
    "name entities explicitly, resolving any bridge entity from the evidence so far. "
    "If the evidence already contains everything needed to answer the main question, "
    "reply with exactly: ENOUGH. Output ONLY the sub-question text or ENOUGH - no prose."
)

def evidence_digest(pools, max_docs=6, max_chars=160):
    # a short, deduped digest of the evidence retrieved so far, to condition the next sub-query.
    seen, lines = set(), []
    for pool in pools:
        for c in pool:
            if c.doc_id in seen:
                continue
            seen.add(c.doc_id)
            lines.append(f"- {c.title}: {(c.text or '')[:max_chars]}")
            if len(lines) >= max_docs:
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(none)"

def next_subquery(question, pools, sub_queries):
    # the LLM reads the evidence so far and asks the next still-missing sub-question (or ENOUGH).
    user = (
        f"Main question: {question}\n\n"
        f"Evidence so far:\n{evidence_digest(pools)}\n\n"
        "Sub-questions already asked:\n" + ("\n".join(f"- {s}" for s in sub_queries) or "(none)") +
        "\n\nNext sub-question (or ENOUGH):"
    )
    text = ask_llm(IRCOT_SYSTEM, user, max_tokens=80)
    t = (text or "").strip()
    if not t or t.upper().startswith("ENOUGH"):
        return None
    return re.sub(r"^[\-\d\.\)\s]+", "", t.splitlines()[0]).strip() or None
""")
md(r"""
Watch what one retrieve can and cannot reach on the Jim Wilson question:

> *What sea washes on the shores of the birthplace of Jim Wilson?*

- **Hop 1 (the birthplace):** retrieve the question as written -> County Antrim
- **Hop 2 (the sea):** the decomposer reads Hop 1 and asks the still-missing question -> Irish Sea
""")
code(r"""
multi = by_id["2hop__615262_131886"]
gold = set(multi["gold_doc_ids"])

# HOP 1: retrieve the question as written; look deep (top-100) for the second passage.
hop1 = hybrid_search(multi["question"], limit=100)
print(f"Q: {multi['question']}\n")
print("HOP 1 - retrieve the question as written (top-3):")
show_hits(hop1, gold)

missing = gold - {p.id for p in hop1[:3]}
reachable_deep = bool(missing & {p.id for p in hop1})
print(f"supporting passages in the top-3: {len(gold) - len(missing)}/{len(gold)} "
      f"(only the Hop-1 passage, whose snippet names the birthplace).")
print(f"the missing Hop-2 passage is {'in' if reachable_deep else 'NOT in'} the fused top-100 "
      f"(in neither retriever's top-50): genuinely out of reach, not just mis-ranked.\n")

# HOP 2: let the decomposer write the follow-up (its real LLM step), then retrieve it.
hop2_query = next_subquery(multi["question"], [to_passages(hop1[:TOP_K])], [])
print(f'HOP 2 - the decomposer reads Hop 1 and asks:  "{hop2_query}"')
show_hits(hybrid_search(hop2_query), gold)
""")
md(r"""
So the answer needed **two** retrievals, and the decomposer could only write the second after Hop 1
revealed the bridge entity (County Antrim). That is a **recall** gap, not a ranking one. The
baseline does only the first retrieve, which is why multi-hop is its weak spot. CP3 wraps this
generate-and-retrieve step into the full `decompose()` tier (looping over hops, unioning evidence).
""")

# ============================================================================ CP2
md(r"""
## CP2: confidence signals, reading the retrieval

A **confidence signal** (or weakness signal) is a cheap scalar you read off the retrieval result to
estimate one thing: is this good enough to answer from, or should I spend more? The agent reads it at
tier 1, before it has paid for any correction.

That "before" is the whole constraint. The signal runs on **every** query, ahead of the decision to
escalate, so it has to be **near-instant**: read from what you already retrieved, or at most one
extra cheap query. A signal that costs as much as the fix it gates buys you nothing. So a candidate
earns its place on **two** axes:

1. **Does it separate** good retrievals from weak ones? (its discriminative power, measured by the AUC below)
2. **Is it cheap?** (near-instant, or it defeats the purpose)

We test a handful of candidates on both, on this corpus, and keep only what survives. The selection
is the lesson; the winners are corpus-specific (more on that below).
""")
md(r"""
### The candidates, by family

Five candidates, each reading a different facet of one retrieval:

| family | signal | what it reads | flags weak when | tends to matter on |
|---|---|---|---|---|
| height | `max_score` | top-1 fused score | low | score-calibrated single-retriever setups |
| spread (fused) | `score_variance` | spread of the fused top-k scores | low (flat ranking) | corpora where fusion keeps score range |
| spread (raw) | `dense_variance` | spread of the **raw dense** cosines | low (dense can't separate its hits) | most dense-retrieval corpora |
| coverage | `evidence_coverage` | question entities present in the top-k text | low (text misses them) | entity-lookup / keyword data |
| agreement | `retriever_divergence` | dense vs miniCOIL top-k overlap | high (they disagree) | lexical-mismatch / jargon / OOV data |

**Raw dense vs fused** is the design choice that matters most. Spread can be read on the fused RRF
score or on the raw dense cosines. RRF compresses scores into ranks and throws the spread away, so
the *fused* version is blunt while the *raw-dense* version keeps its dynamic range. We benchmark both,
and the benchmark below crowns the raw-dense one. (We do *not* separately benchmark the rank-1-minus-
rank-K "gap": it measures the same thing as spread and runs ~0.99 correlated with it, so `variance`
stands in for both. That is the easy redundancy. The interesting one is below.)

**Cost is not equal**, and that is the second axis. `max_score`, `score_variance`, and
`evidence_coverage` are free: they read the hybrid result you already have. `dense_variance` and
`retriever_divergence` each cost one extra single-retriever query, reusing embeddings the hybrid query
already computed (no extra model). We read divergence against miniCOIL, the sparse retriever already in
the hybrid query, rather than loading a separate BM25; a maximally-lexical retriever like BM25 is often
a sharper divergence detector, and is what you would reach for on data where this signal earns its
keep.
""")
md(r"""
First, the materials the signals read: we already have the fused hybrid result, so we add the two raw
single-retriever rankings (dense and miniCOIL) and the teaching-simple entity extractor coverage uses.
""")
code(r"""
def dense_ranking(question, enc=None):
    # the RAW dense cosine ranking (a dense-only Qdrant query), pre-fusion: [(doc_id, cosine), ...].
    dense, _minicoil = enc or embed(question)
    pts = client.query_points(COLLECTION, query=dense, using=DENSE_VEC, limit=TOP_K, with_payload=False).points
    return [(p.id, p.score) for p in pts]

def minicoil_ranking(question, enc=None):
    # the RAW miniCOIL ranking (a sparse-only Qdrant query), pre-fusion: [doc_id, ...]. Reuses the
    # miniCOIL embedding the hybrid query already computes, so it costs one query, not a new model.
    _dense, minicoil = enc or embed(question)
    pts = client.query_points(
        COLLECTION,
        query=models.SparseVector(indices=minicoil.indices.tolist(), values=minicoil.values.tolist()),
        using=MINICOIL_VEC, limit=TOP_K, with_payload=False).points
    return [p.id for p in pts]

_QUESTION_STOP = {"what", "who", "whom", "whose", "where", "when", "which", "why", "how",
                  "is", "was", "are", "were", "did", "do", "does", "the", "a", "an", "name",
                  "in", "of", "on", "at", "to", "for", "by", "as", "that", "this"}
_NAME_CONNECTORS = {"of", "the", "and", "de", "von", "van", "del", "la", "el", "da", "di", "&"}

def question_entities(question):
    # teaching-simple entity extractor: maximal runs of Capitalized tokens (joined by lowercase
    # connectors like 'of'/'the'), plus 4-digit years; drop the sentence-initial question word.
    # The production upgrade is spaCy NER (mentioned in the docs, not a live dependency here).
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
            cur.append(w)
        elif cur:
            ents.append(" ".join(cur)); cur = []
    if cur:
        ents.append(" ".join(cur))
    out = {e.lower() for e in ents if len(e) >= 2}
    out |= set(re.findall(r"\b\d{4}\b", question or ""))
    return out
""")
md(r"""
Now the five candidates, one line each (one row of the table above). `retrieve_signals` runs the
three reads once. Note the split the table flagged: `dense_variance` reads the **raw dense** cosines;
the rest read the **fused** scores or the rankings.
""")
code(r"""
def retrieve_signals(question, enc=None):
    # one encode, three reads: the fused hybrid result + the two raw single-retriever rankings.
    enc = enc or embed(question)
    return hybrid_search(question, enc=enc), dense_ranking(question, enc=enc), minicoil_ranking(question, enc=enc)

def max_score(fused):       return fused[0].score                                  # height: top-1 fused score
def score_variance(fused):  return statistics.pstdev([p.score for p in fused])     # spread of the FUSED scores
def dense_variance(dense):  return statistics.pstdev([s for _, s in dense])        # spread of the RAW DENSE cosines

def evidence_coverage(question, fused):                                            # coverage: question entities present?
    ents = question_entities(question)
    if not ents:
        return 1.0
    blob = " ".join(f"{p.payload['title']} {p.payload['text']}" for p in fused).lower()
    return sum(1 for e in ents if e in blob) / len(ents)

def retriever_divergence(dense, sparse_ids):                                       # agreement: dense vs miniCOIL disjointness
    dense_ids = [i for i, _ in dense]
    return 1.0 - len(set(dense_ids) & set(sparse_ids)) / max(len(dense_ids), len(sparse_ids))
""")
md(r"""
**How we pick which signals to keep, computed live.** A signal earns its place only if it predicts a
weak retrieval, where "weak" means the supporting set is *not* all in the top-3. We know which
retrievals are weak here because this is calibration data with gold labels; at run time we have only
the signals, which is the whole reason to benchmark them against a golden set first. We score each
candidate by its **AUC** at separating good retrievals from weak ones (0.5 = chance, 1.0 = perfect),
using `max(auc, 1 - auc)` so the direction does not matter, over the frozen calibration split right
here (about 150 questions, Qdrant-only, no LLM, roughly a minute). The benchmark below is the real
thing, not a cached table.
""")
code(r"""
# The frozen calibration split (seed 5252) only picks WHICH ~150 questions to score; every signal
# value and label below is computed LIVE. The full_gold@3 label = all gold docs inside the fused top-3.
manifest = json.loads((REPO / "artifacts/mixed_manifest.json").read_text())
cal_sel = manifest["splits"]["calibration"]
cal_ids = [*cal_sel["single"], *cal_sel["multi"], *cal_sel["unanswerable"]]
cal_questions = [by_id[i] for i in cal_ids if by_id[i].get("answerable") and by_id[i].get("gold_doc_ids")]

def feature_row(q):
    fused, dense, sparse_ids = retrieve_signals(q["question"])
    gold = set(q["gold_doc_ids"])
    return {
        "full_gold_label": 1 if gold.issubset({p.id for p in fused[:ANSWER_K]}) else 0,
        "dense_variance": dense_variance(dense), "score_variance": score_variance(fused),
        "max_score": max_score(fused), "evidence_coverage": evidence_coverage(q["question"], fused),
        "divergence_minicoil": retriever_divergence(dense, sparse_ids),
    }

calibration = [feature_row(q) for q in cal_questions]   # ~1 min, no LLM
print(f"benchmarked {len(calibration)} calibration questions live; "
      f"{sum(r['full_gold_label'] for r in calibration)} good / "
      f"{sum(1 - r['full_gold_label'] for r in calibration)} weak retrievals")
""")
md(r"""
The AUC catalog, computed from those live features:
""")
code(r"""
labels = [r["full_gold_label"] for r in calibration]

SIGNAL_COLUMN = {   # signal -> the feature-matrix column it reads (divergence is stored per detector)
    "dense_variance": "dense_variance", "score_variance": "score_variance",
    "max_score": "max_score", "evidence_coverage": "evidence_coverage",
    "retriever_divergence": "divergence_minicoil",
}

def signal_auc(name):
    raw = roc_auc_score(labels, [r[SIGNAL_COLUMN[name]] for r in calibration])
    return max(raw, 1 - raw)            # discriminative power, regardless of direction

aucs = {name: signal_auc(name) for name in SIGNAL_COLUMN}
kept = {"dense_variance", "score_variance"}    # cleared the AUC bar AND held up on a held-out validation split

catalog = pd.DataFrame([
    {"signal": name, "AUC": aucs[name], "verdict": "kept" if name in kept else "dropped"}
    for name in sorted(aucs, key=lambda n: -aucs[n])
])
catalog
""")
md(r"""
Three of the five drop out, for two different reasons. `retriever_divergence` and `evidence_coverage`
never clear the AUC bar. `max_score` clears it, but it turns out ~0.92 correlated with `score_variance`
on this corpus, so the two are near-duplicates and we keep the stronger one. That second cut is the one
you could not predict in advance: height and spread are distinct ideas that just happen to move
together here. Both cuts, from the same live features:
""")
code(r"""
def abs_corr(a, b):
    return abs(float(np.corrcoef([r[SIGNAL_COLUMN[a]] for r in calibration],
                                 [r[SIGNAL_COLUMN[b]] for r in calibration])[0, 1]))

below_bar = sorted((n for n in aucs if aucs[n] < 0.62), key=lambda n: -aucs[n])
print("below the AUC bar (< 0.62), dropped:")
for n in below_bar:
    print(f"  {n:22s} AUC {aucs[n]:.2f}")

print("\nredundant (|corr| > 0.85 with a signal we keep), dropped:")
print(f"  {'max_score':16s} |corr| {abs_corr('max_score', 'score_variance'):.2f} with score_variance")
print(f"\nkept (distinct and discriminative): {sorted(kept)}")
""")
md(r"""
The same benchmark as a picture: each signal's value on good vs weak retrievals. Where the two boxes
pull apart, it separates (high AUC); where they overlap, it does not.
""")
code(r"""
plot_signal_separation(calibration, aucs, kept, SIGNAL_COLUMN)
""")
md(r"""
Two survivors, `dense_variance` and `score_variance`, and they are *not* redundant with each other
(raw-dense vs fused spread correlate only ~0.5), so each earns its place. One discipline to carry over:
we select the signals on a held-out validation split and calibrate the floor below on calibration,
never touching test, so the choice is not just fit to the numbers you happen to see here. The deeper
lesson: **the winners are corpus-specific.** On your data the AUCs land differently, a signal weak
here may be strong, and a different set may survive. The method transfers, the selection does not.
""")

# ============================================================================ the gate
md(r"""
## The gate: turning the signals into a decision

Two kept signals, `dense_variance` and `score_variance`. Using them is two steps: **tune** where each
floor sits, then **wire** the gate the loop calls.

### Tuning the floor

Each signal fires when it drops below a floor. We pick that floor live, at the point that maximizes
Youden's J (true-positive rate minus false-positive rate) on the calibration features. Lower the
floor and you escalate less often (higher precision, lower recall); raise it and you catch more weak
retrievals at the cost of escalating some healthy ones.
""")
code(r"""
weak = [r["full_gold_label"] == 0 for r in calibration]    # True = retrieval was weak (gold not all in top-3)

def youden_floor(values):
    # the "fires below" floor that maximizes Youden's J = TPR - FPR over the calibration features.
    best = None
    for thr in sorted(set(values)):
        pred = [v < thr for v in values]
        tp = sum(1 for p, w in zip(pred, weak) if p and w)
        fp = sum(1 for p, w in zip(pred, weak) if p and not w)
        fn = sum(1 for p, w in zip(pred, weak) if not p and w)
        tn = sum(1 for p, w in zip(pred, weak) if not p and not w)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if best is None or (tpr - fpr) > best[1]:
            best = (thr, tpr - fpr)
    return best[0]

DV_FLOOR = youden_floor([r["dense_variance"] for r in calibration])
SV_FLOOR = youden_floor([r["score_variance"] for r in calibration])
print(f"calibrated floors:  dense_variance < {DV_FLOOR:.5f}   score_variance < {SV_FLOOR:.5f}\n")

# the operating point of the primary floor, read off the same calibration features
fires = [r["dense_variance"] < DV_FLOOR for r in calibration]
tp = sum(f and w for f, w in zip(fires, weak)); fp = sum(f and not w for f, w in zip(fires, weak))
fn = sum((not f) and w for f, w in zip(fires, weak))
print("at the dense_variance floor:")
print(f"  precision       = {tp / (tp + fp):.3f}   (of the queries we escalate, how many were truly weak)")
print(f"  recall          = {tp / (tp + fn):.3f}   (of the truly weak queries, how many we catch)")
print(f"  escalation rate = {sum(fires) / len(fires):.2f}   (fraction of queries sent past tier 1)")
""")
md(r"""
The same floor as a picture. The left panel shows where it sits on the good vs weak distributions
(escalate to its left); the right panel sweeps it, so you can read precision, recall, and escalation
at any floor, not just the one we chose. `score_variance` works the same way.
""")
code(r"""
plot_gate(calibration, DV_FLOOR)
""")
md(r"""
### The implementation

The gate the loop calls: escalate if **either** kept signal is below its floor. Its whole cost on top
of the hybrid retrieve you already run is a single dense-only query, exactly the near-instant budget
we set out to keep.
""")
code(r"""
def retrieval_is_weak(question):
    # the tier-1 gate: escalate if EITHER kept signal is below its calibrated floor.
    enc = embed(question)
    dv = dense_variance(dense_ranking(question, enc=enc))
    sv = score_variance(hybrid_search(question, enc=enc))
    return dv < DV_FLOOR or sv < SV_FLOOR
""")

# ============================================================================ CP3
md(r"""
## CP3: the corrective tiers, run live

Each tier is matched to a failure mode. A weak **single-hop** lookup is mis-ranked (a *precision*
problem); a weak **multi-hop** query is missing a hop (a *recall* problem).

### Tier 2: ColBERT late interaction (single-hop precision)

ColBERT scores a query against a passage **token by token** (MaxSim over Qdrant native
multivectors), catching term-level matches a single pooled embedding blurs away. Here is the actual
Qdrant call: prefetch a dense pool, then rescore it with the ColBERT multivector.
""")
code(r"""
def colbert_rerank(question, limit=TOP_K):
    # Qdrant native multivector: prefetch a dense pool, rescore with ColBERT MaxSim (late interaction).
    dense, _minicoil = embed(question)
    colbert_vecs = [v.tolist() for v in next(iter(colbert_model.query_embed(question)))]
    return client.query_points(
        COLBERT_COLLECTION,
        prefetch=[models.Prefetch(query=dense, using=DENSE_VEC, limit=RETRIEVE_N)],
        query=colbert_vecs, using=COLBERT_VEC, limit=limit, with_payload=True,
    ).points

tier2 = by_id["2hop__82744_23140__h0"]            # a weak single-hop lookup
gold = tier2["gold_doc_ids"]
print(f"Q: {tier2['question']}\n")
print("hybrid retrieve, supporting passage buried:")
show_hits(hybrid_search(tier2["question"]), gold)
print("\nColBERT late interaction, the right passage promoted:")
show_hits(colbert_rerank(tier2["question"]), gold)
""")
md(r"""
Hybrid matched the entity "Nigeria" and buried the passage that actually answers the question (the
GDP table); ColBERT pulled it to rank 1. Across the validation set ColBERT is a wash *on average*
(single-hop is already mostly solved), but on the weak lookups the gate routes to it, it lifts
precision. A cross-encoder reranker ties ColBERT here; we use ColBERT because it is a native Qdrant
multivector. On your data, test both.

### Tier 3: decompose (IRCoT) for multi-hop recall

No reranking can fix a *missing* hop. In CP1 you watched the decomposer generate one follow-up and
reach the missing passage; `decompose()` wraps that into a loop: retrieve, ask the next still-missing
sub-question (the CP1 `next_subquery`), retrieve that, and union the evidence, until the LLM says
ENOUGH. Here is the full loop, run live on the same Jim Wilson question.
""")
code(r"""
def union_pool(pools, k=TOP_K):
    # merge the per-hop passage pools, keeping the MAX score per doc, take the top-k. This is what
    # lets a later hop's strongest passage surface into the final set (the missing-hop recovery).
    best = {}
    for pool in pools:
        for c in pool:
            if c.doc_id not in best or c.score > best[c.doc_id].score:
                best[c.doc_id] = c
    return sorted(best.values(), key=lambda c: c.score, reverse=True)[:k]

def decompose(question, max_hops=4):
    # IRCoT: retrieve, ask the next missing sub-question, retrieve, union. Repeat until ENOUGH.
    pools = [to_passages(hybrid_search(question))]
    sub_queries = []
    for _ in range(max_hops - 1):
        next_q = next_subquery(question, pools, sub_queries)
        if next_q is None:
            break
        sub_queries.append(next_q)
        pools.append(to_passages(hybrid_search(next_q)))
    return union_pool(pools, TOP_K), sub_queries

multi = by_id["2hop__615262_131886"]
gold = multi["gold_doc_ids"]
print(f"Q: {multi['question']}\n")
print("hybrid retrieve (single pass), only the first hop is reachable:")
show_hits(hybrid_search(multi["question"]), gold)

pool, sub_queries = decompose(multi["question"])
print("\ndecompose reads hop 1, then asks the still-missing sub-question:")
for sub_question in sub_queries:
    print(f"  -> {sub_question}")
print("\nunioned evidence, the second supporting passage now in context:")
show_hits(pool, gold, k=4)
""")
md(r"""
The hybrid pass found the Jim Wilson passage (its snippet names County Antrim); decompose asked the
follow-up and pulled in the passage about its sea. Now the aggregate, measured across the validation
set with a per-query counterfactual. This table comes from the **offline eval** (every policy over
the whole workload with the LLM, ~45 min, run once), the first of four precomputed scorecards.
""")
code(r"""
policy_comparison = load_artifact("policy_comparison_val.json")   # offline eval, run once
overall = policy_comparison["overall"]
multi_hop = policy_comparison["by_type"]["multi_hop"]

pd.DataFrame([
    {"policy": "always-answer (baseline)", "single-hop recall@3": overall["always_answer"]["recall@3"],
     "single-hop MRR": overall["always_answer"]["mrr"], "multi-hop full_gold@3": multi_hop["always_answer"]["full_gold@3"]},
    {"policy": "always-ColBERT", "single-hop recall@3": overall["always_colbert"]["recall@3"],
     "single-hop MRR": overall["always_colbert"]["mrr"], "multi-hop full_gold@3": multi_hop["always_colbert"]["full_gold@3"]},
    {"policy": "always-decompose", "single-hop recall@3": overall["always_decompose"]["recall@3"],
     "single-hop MRR": overall["always_decompose"]["mrr"], "multi-hop full_gold@3": multi_hop["always_decompose"]["full_gold@3"]},
])
""")

# ============================================================================ assembled loop
md(r"""
## The assembled loop: `solve()`

Now put it together. The whole agent is this one function: retrieve, read the signal, climb only as
far as the query needs. Every piece below is something you built above, including `generate_answer`
(the grounded answer step, defined here), which reads only the focused top-3 and self-abstains with
`INSUFFICIENT_CONTEXT` when the evidence is not enough.
""")
code(r"""
ANSWER_SYSTEM = (
    "You answer a question using ONLY the numbered context passages provided. "
    "Reply with ONLY the final answer on a single line: a name, date, number, or short "
    "noun phrase, usually one to six words. Do NOT show reasoning or steps, do NOT "
    "restate the question, do NOT write 'I need to find', do NOT explain. Output just "
    "the answer text. If the passages do not contain the information needed to answer, "
    "reply with exactly: INSUFFICIENT_CONTEXT"
)

def generate_answer(question, passages):
    # Claude answers grounded in the focused top-ANSWER_K passages, or emits INSUFFICIENT_CONTEXT.
    if not passages:
        return "INSUFFICIENT_CONTEXT"
    ctx = "\n".join(f"[{i}] {c.title}. {(c.text or '')[:700]}"
                    for i, c in enumerate(passages[:ANSWER_K], 1))
    text = ask_llm(ANSWER_SYSTEM, f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer (answer only):",
                   max_tokens=150)
    t = (text or "").strip()
    if "INSUFFICIENT_CONTEXT" in t.upper() or not t:
        return "INSUFFICIENT_CONTEXT"
    last = [ln.strip() for ln in t.splitlines() if ln.strip()][-1]
    return re.sub(r"^(answer|the answer is|final answer)[:\-\s]+", "", last, flags=re.I).strip()

def looks_multi_hop(question):
    # cheap, gold-free router: >= 2 named entities or a long question -> likely a missing hop.
    return len(question_entities(question)) >= 2 or len(question.split()) >= 12

def solve(question):
    # the self-correcting retrieval loop: read the signal, climb only as far as the query needs.
    if not retrieval_is_weak(question):
        return to_passages(hybrid_search(question)), "tier 1: confident, answer from the hybrid top-3"
    if looks_multi_hop(question):
        pool, sub_queries = decompose(question)
        return pool, f"tier 3: weak + multi-hop, decomposed ({len(sub_queries)} sub-question(s))"
    return to_passages(colbert_rerank(question)), "tier 2: weak single-hop, ColBERT late-interaction"
""")
md(r"""
Three queries, three paths from one agent. Each answer is generated from the routed top-3, so this
runs the full loop live.
""")
code(r"""
routing_demos = [
    ("confident single-hop", "2hop__101521_42157__h0"),
    ("weak single-hop",      "2hop__130545_45439__h0"),
    ("multi-hop",            "2hop__615262_131886"),
]
for label, qid in routing_demos:
    q = by_id[qid]
    pool, route = solve(q["question"])
    answer = generate_answer(q["question"], pool)
    print(f"[{label}]")
    show_run(q["question"], route, answer, pool, q["gold_doc_ids"])
""")
md(r"""
The cheap query answered at tier 1, the weak lookup escalated to ColBERT, the multi-hop decomposed.
The signal decides how far to climb, so each query pays only for the correction it needs. Now the
cost/quality frontier across the validation set (cost = mean LLM sub-query calls per query;
decompose is the expensive tier; ColBERT adds a retrieval pass, not an LLM call, so it reads ~0
here).
""")
code(r"""
frontier_validation = frontier_table(overall, mrr_key="mrr", cost_key="cost_llm")
frontier_validation
""")
md(r"""
Read this as a cost/quality tradeoff, not a single winner. The ladder reaches about the same
answerable quality as always-decompose at **under half** its LLM cost. It **leads on MRR**;
always-decompose **leads on full_gold@3** (completeness). The ladder does not dominate: it is the
efficient point.
""")

# ============================================================================ STOP
md(r"""
## The STOP decision: a smaller, separate lever

Stopping is a different decision from routing: whether to answer at all or abstain. The **gentle
stop** is the default (the generator answers, or says it lacks enough), and it keeps the most
answers. For workloads where abstaining out of caution beats a confident wrong answer, swap in an
**LLM sufficiency check**: it catches far more unanswerables but over-refuses some answerables. Here
is that autorater, one fast-model call.
""")
code(r"""
SUFFICIENCY_SYSTEM = (
    "You judge whether the provided context contains ENOUGH information to answer the "
    "question with certainty. First decompose the question into the facts it requires; "
    "the context is SUFFICIENT only if EVERY required fact is explicitly present in the "
    "context. Use ONLY the context, not outside knowledge. If any required fact is "
    'missing, it is insufficient. Reply ONLY with compact JSON: {"sufficient": true|false}.'
)

def sufficiency_judge(question, passages):
    # the STOP autorater: does the retrieved context contain every fact the question requires?
    if not passages:
        return False
    ctx = "\n".join(f"[{i}] {c.title}. {(c.text or '')[:600]}" for i, c in enumerate(passages, 1))
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nIs the context sufficient to answer the question?"
    try:
        text = ask_llm(SUFFICIENCY_SYSTEM, user, max_tokens=512, model=FAST_MODEL)
        m = re.search(r'"?sufficient"?\s*[:=]\s*"?(true|false)"?', text, re.I)
        if m:
            return m.group(1).lower() == "true"
        return bool(json.loads(text[text.find("{"): text.rfind("}") + 1]).get("sufficient"))
    except Exception:    # default to sufficient (do not over-abstain) on a parse/API failure
        return True
""")
md(r"""
The two stops side by side across the workload (the second precomputed **offline-eval** scorecard).
"Full workload handled" counts a query as handled when the agent answers correctly OR correctly
refuses an unanswerable.
""")
code(r"""
stop_variants = load_artifact("targeted_stop_v25.json")["variants"]   # offline eval, run once
stop_rows = [
    ("hybrid baseline + gentle",       "baseline_hybrid_gentle"),
    ("ladder + gentle (default)",      "ladder_gentle"),
    ("ladder + LLM sufficiency check", "ladder_autorater_all"),
]
pd.DataFrame([
    {"setup": label,
     "catches unanswerable": stop_variants[key]["abstain_unans"],
     "over-refuses answerable": stop_variants[key]["false_stop_ans"],
     "full workload handled": stop_variants[key]["selective_accuracy"]}
    for label, key in stop_rows
])
""")
md(r"""
Routing is not stopping: a good router is not automatically a good stopper, and the ceiling on either
is retrieval completeness. Here are both stops on an unanswerable, run live: the agent decomposes,
the gentle stop self-abstains, and the autorater independently judges the context insufficient.
""")
code(r"""
unanswerable = by_id["2hop__108098_170204"]
pool, route = solve(unanswerable["question"])
gentle = generate_answer(unanswerable["question"], pool)
sufficient = sufficiency_judge(unanswerable["question"], pool[:ANSWER_K])
show_run(unanswerable["question"], route, gentle, pool, [])
print(f"  autorater sufficiency check: {'sufficient' if sufficient else 'insufficient -> abstain'}")
""")

# ============================================================================ WRAP
md(r"""
## Wrap: the honest scorecard (held-out test)

The adaptive ladder against the fixed policies on the test slice (the third precomputed
**offline-eval** scorecard). We lead with retrieval precision (the contamination-resistant measure of
what the loop fixes) and report answer quality with a semantic judge, not exact match. Honest caveat:
this test slice partly reuses questions from earlier rounds (disclosed in `headline_final_v25.json`),
so treat it as held-out from threshold tuning, not as a pristine never-seen set.
""")
code(r"""
headline = load_artifact("headline_final_v25.json")    # offline eval, run once
frontier_test = frontier_table(headline["overall"], mrr_key="mrr_first", cost_key="llm_calls")
frontier_test
""")
md(r"""
Answer quality uses a **gpt-5.5 semantic judge** (it credits correct-but-paraphrased answers), not
exact match (the fourth precomputed **offline-eval** scorecard).
""")
code(r"""
judge = load_artifact("judge_eval_v25.json")    # offline eval, run once
by_policy = judge["by_policy"]
answer_quality = pd.DataFrame([
    {"policy": name.replace("_", "-"),
     "overall": by_policy[name]["overall"]["judge"],
     "single-hop": by_policy[name]["single_hop"]["judge"],
     "multi-hop": by_policy[name]["multi_hop"]["judge"]}
    for name in ("always_answer", "ladder", "always_decompose")
])
answer_quality
""")
md(r"""
### What we learned (and what we honestly did not)

- **Adaptive routing is cost-efficient, not dominant.** The ladder reaches near-decompose answerable
  quality at about 40% of always-decompose's LLM cost and beats the no-correction baseline on
  retrieval. It leads MRR; always-decompose leads full_gold@3. It wins the cost/quality tradeoff, not
  every metric.
- **The gains are per-slice.** Decomposition lifts multi-hop full_gold and multi-hop answer quality
  by tens of points; the overall lift is small because the easy single-hop majority is already
  strong. That dilution *is* the cost story: easy queries stay cheap.
- **The right substrate makes a weak signal strong.** Reading spread on raw dense cosines, not the
  fused score, is what made the confidence gate work.
- **Routing is not stopping.** A real sufficiency check wins the full workload (about 0.63 vs the
  baseline's 0.53) but trades away some answers.
- **Honesty on decompose.** IRCoT's sub-queries are written by an LLM that may know the bridge
  entities, so its lift is an upper bound that shrinks on unseen corpora. Recalibrate on yours.

**What we tested that did NOT win here (but wins elsewhere):** `evidence_coverage` and
`retriever_divergence` (useful on single-hop / entity-lookup / lexical-mismatch data); `max_score`
(a calibrated single-retriever QPP signal); a cross-encoder reranker (tied ColBERT here); and the
recall@10 framing (it hid all the headroom, since single-hop is ~98% solved at top-10). None is
useless: each would win on a different workload.
""")
md(r"""
### How to adapt this to your workflow

The numbers are ours; the method is yours. To build a self-correcting loop on your own corpus:

1. **Set your answer-context size.** Decide how many passages the LLM reads (we used top-3). That
   choice is what makes ranking precision matter and gives the tiers room to work.
2. **Build candidate signals and validate them.** Compute cheap query-time readings, score each
   against your golden set (AUC), drop the redundant twins, keep what separates on your data. Select
   on one split, calibrate the threshold on another, never touch test.
3. **Match each tier to a real failure mode.** Inspect your own traces, find your failure modes,
   and pick the cheapest action that fixes each.
4. **Measure on cost AND quality, per query.** Run every policy on every query, read the frontier,
   keep the loop only where it earns its cost.
5. **Choose your stop.** Gentle by default; an LLM sufficiency check when a confident wrong answer
   is more expensive than an honest abstention.

Take home: this notebook and the reusable `src/` modules (the same logic, packaged).
""")

# ============================================================================ write
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
