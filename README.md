# Building Self-Correcting Agentic Retrieval Loops

A hands-on workshop. You start from a prebuilt Qdrant-powered retrieval agent and
make it **self-correcting**: it retrieves once, evaluates the evidence with cheap
signals, then invokes more expensive retrieval steps only when needed.

```
Tier 1  answer      - confident? answer now from a focused top-3 (the cheap path)
Tier 2  ColBERT     - weak single-hop lookup? a token-level precision re-retrieval
Tier 3  decompose   - weak multi-hop? recover the missing hop (IRCoT)
        then ANSWER or STOP (a separate sufficiency decision)
```

The lesson is a **method**, not a recipe: define what good retrieval means, build cheap
signals, validate which ones predict weak evidence on *your* data, route each query to
the cheapest sufficient action, and measure whether it helped once **cost and latency**
are counted. The lab uses a **mixed workload**: single-hop, multi-hop, and
unanswerable questions.

- **The lab:** [`notebooks/lab.ipynb`](notebooks/lab.ipynb) - CP1 concepts/baseline, CP2 metrics/signals/gate, CP3 corrective loop + STOP, then wrap.
- **Intro deck outline:** [`deck/intro_outline.md`](deck/intro_outline.md).
- **Results summary:** [`briefing.html`](briefing.html).
- **Docs tutorial outline:** [`tutorials/in-loop-evals-OUTLINE.md`](tutorials/in-loop-evals-OUTLINE.md).

On the workshop VM everything below is pre-installed, pre-embedded, and warm: no
setup in the room. These instructions reproduce that state from a clean clone.

## Prerequisites

- Docker (for Qdrant)
- Python 3.12
- An `.env` at the repo root with `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`

## Quickstart (build the VM state from a clean clone)

```bash
# 1. Qdrant
docker compose up -d                                  # qdrant/qdrant:v1.18.0 on :6333 / :6334

# 2. Python env
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip uv
.venv/bin/uv pip install -r requirements.lock.txt     # exact pins; or requirements.txt for loose floors

# 3. Build the two collections from the committed corpus (one-time; ~22.8k docs).
#    First run downloads + caches the FastEmbed models, so it takes a few minutes.
.venv/bin/python scripts/setup_collections.py         # musique: dense + bm25 + miniCOIL
.venv/bin/python scripts/setup_colbert.py             # musique_colbert: dense (reused) + ColBERT multivector

# 4. Verify the VM is "Ready"
.venv/bin/python scripts/check_env.py --llm
```

`check_env.py` confirms the libraries import, FastEmbed exposes the models the lab
uses, Qdrant is healthy, **both collections are populated**, and (with `--llm`) the
Claude agent and the gpt-5.5 judge both answer. Then open `notebooks/lab.ipynb` and
run the Setup cell; it should print `Ready`.

## The stack

| Layer | Choice |
|---|---|
| Vector DB | Qdrant 1.18.0 (Docker), two collections, named vectors |
| Dense | `BAAI/bge-base-en-v1.5` (768-d, cosine), FastEmbed / local ONNX |
| Sparse | `Qdrant/minicoil-v1` (baseline fusion sparse) + `Qdrant/bm25` (divergence-detector candidate) |
| Fusion | Reciprocal Rank Fusion (RRF), server-side via the Qdrant Query API - **no cross-encoder in the baseline** |
| Tier-2 fix | `answerdotai/answerai-colbert-small-v1` ColBERT late interaction (Qdrant native multivector, MaxSim) |
| Reranker | `jinaai/jina-reranker-v2-base-multilingual` cross-encoder - a *measured alternative* to the ColBERT rung, not baseline |
| Agent | Claude Sonnet 4.6 via LiteLLM (decompose + answer) |
| Stop autorater | Claude Haiku 4.5 via LiteLLM (the optional LLM sufficiency check) |
| Judge | gpt-5.5 (cross-provider, reduces self-preference bias; eval only) |
| IR metrics | ranx |
| Dataset | MuSiQue, recast as a mixed single-hop / multi-hop / unanswerable workload |

All embedding and reranking is local (FastEmbed ONNX), so query-time encoding is free
and offline; only the agent, the stop autorater, and the judge hit the network.

## The mixed workload and answer context

The agent answers from a **top-3 answer context**: the LLM only reads the first three
retrieved passages. That makes ranking quality visible. In the notebook, "good
retrieval" means the needed supporting passages land in that top-3 window; weak
retrieval means the answer context is missing needed evidence. Native unanswerables
drive the STOP decision.

## Reproducibility

Everything needed to rebuild is in the repo:

- `data/` - the corpus (`corpus.jsonl`, 22,808 passages), the question splits
  (`questions.jsonl`), the derived mixed workload (`questions_mixed.jsonl`), and the
  dataset metadata. The collections in step 3 are built directly from these files.
- `artifacts/mixed_manifest.json` - the **frozen** mixed-workload population (seeded;
  every question id per split is pinned), so the eval set is reproducible by id.
- `artifacts/*.json` - frozen manifests and precomputed workload-level scorecards used
  by the STOP and Wrap sections. The signal benchmark in the notebook runs live over
  the calibration split.

To regenerate the **dataset itself** from MuSiQue (needs HuggingFace access):
`scripts/prepare_data.py` then `scripts/prepare_mixed.py`. To regenerate the eval
artifacts: the `scripts/run_*.py` and `scripts/calibrate_mixed.py` (these call the
agent/judge LLMs).

## Repo layout

```
notebooks/lab.ipynb    # the lab: CP1 concepts -> CP2 gate -> CP3 loop/STOP -> wrap
docker-compose.yml     # Qdrant
requirements.txt       # loose deps; requirements.lock.txt has exact pins
src/                   # reusable modules: config, data, retrieval, signals, policy, agent, trace, eval
scripts/               # data build, collection setup, env check, eval + deliverable generators
data/                  # corpus, splits, derived mixed workload
artifacts/             # precomputed eval artifacts + frozen manifest the notebook reads
```

Modules under `src/` are imported by bare name; scripts and the notebook put `src/` on
`sys.path` first.
