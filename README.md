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

## Prerequisites

- Docker (for Qdrant)
- Python 3.12
- An `.env` at the repo root with `ANTHROPIC_API_KEY`

## Setup

```bash
# 1. Qdrant
docker compose up -d                                  # qdrant/qdrant:v1.18.0 on :6333 / :6334

# 2. Python env
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip uv
uv pip install -r requirements.lock.txt               # exact pins; or requirements.txt for loose floors

# 3. Build the two collections from the committed corpus (one-time; ~22.8k docs).
#    First run downloads + caches the FastEmbed models, so it takes a few minutes.
python scripts/setup_collections.py                   # musique: dense + miniCOIL
python scripts/setup_colbert.py                       # musique_colbert: dense (reused) + ColBERT multivector

# 4. Verify everything is "Ready"
python scripts/check_env.py --llm
```

`check_env.py` confirms the libraries import, FastEmbed exposes the models the lab
uses, Qdrant is healthy, **both collections are populated**, and (with `--llm`) the
Claude agent answers. Then open `notebooks/lab.ipynb` and run the Setup cell; it
should print `Ready`.

## The stack

| Layer | Choice |
|---|---|
| Vector DB | Qdrant 1.18.0 (Docker), two collections, named vectors |
| Dense | `BAAI/bge-base-en-v1.5` (768-d, cosine), FastEmbed / local ONNX |
| Sparse | `Qdrant/minicoil-v1` (word-sense-aware sparse, IDF modifier) - the baseline fusion sparse |
| Fusion | Reciprocal Rank Fusion (RRF), server-side via the Qdrant Query API - **no cross-encoder in the baseline** |
| Tier-2 fix | `answerdotai/answerai-colbert-small-v1` ColBERT late interaction (Qdrant native multivector, MaxSim) |
| Agent | Claude Sonnet 4.6 via LiteLLM (decompose + answer) |
| Stop autorater | Claude Haiku 4.5 via LiteLLM (the optional LLM sufficiency check) |
| Dataset | MuSiQue, recast as a mixed single-hop / multi-hop / unanswerable workload |

All embedding and reranking is local (FastEmbed ONNX), so query-time encoding is free
and offline; only the agent and the stop autorater hit the network.

## The mixed workload and answer context

The agent answers from a **top-3 answer context**: the LLM only reads the first three
retrieved passages. That makes ranking quality visible. In the notebook, "good
retrieval" means the needed supporting passages land in that top-3 window; weak
retrieval means the answer context is missing needed evidence. Native unanswerables
drive the STOP decision.

## Reproducibility

Everything needed to run the lab is in the repo:

- `data/` - the corpus (`corpus.jsonl`, 22,808 passages), the question splits
  (`questions.jsonl`), the derived mixed workload (`questions_mixed.jsonl`), and the
  dataset metadata. The collections in step 3 are built directly from these files.
- `artifacts/mixed_manifest.json` - the **frozen** mixed-workload population (seeded;
  every question id per split is pinned), so the eval set is reproducible by id.
- `artifacts/{headline_final_v25,targeted_stop_v25}.json` - the precomputed
  workload-level scorecards the Wrap and STOP sections read, committed as frozen
  evidence. The signal benchmark in the notebook itself runs live over the calibration
  split.

The notebook builds retrieval, the signals, the gate, IRCoT, and STOP **inline from
primitives** - lift any cell to reproduce the method or adapt it to your own corpus. To
rebuild the **dataset** from MuSiQue (needs HuggingFace access): `scripts/prepare_data.py`,
then `scripts/audit_single_hop.py`, then `scripts/prepare_mixed.py`.

## Repo layout

```
notebooks/lab.ipynb    # the lab (built inline from primitives): CP1 concepts -> CP2 gate -> CP3 loop/STOP -> wrap
deck/intro_outline.md  # workshop intro deck outline
docker-compose.yml     # Qdrant
requirements.txt       # loose deps; requirements.lock.txt has exact pins
src/                   # config, data, labkit (constants/loaders + notebook rendering)
scripts/               # dataset build, collection setup, env check
data/                  # corpus, splits, derived mixed workload
artifacts/             # the frozen manifest + scorecards the notebook reads
```
