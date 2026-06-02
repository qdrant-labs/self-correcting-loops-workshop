"""Central configuration: model ids, collection schema, retrieval constants, paths.

Single source of truth imported by every module, script, and notebook cell. Keep
magic strings and numbers here so the lab has one place to look and one place to
change. Thresholds are NOT here: they are data-derived per dataset (calibration
split) and live in the calibrated artifacts, never hard-coded as portable.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# --- Qdrant ------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "musique"
# Tier-2 ColBERT showcase lives in a SEPARATE collection (dense + colbert multivector)
# so the validated Tier-1 `musique` collection is never disturbed.
COLBERT_COLLECTION = "musique_colbert"

# --- Embedding / rerank models (FastEmbed, local ONNX, cached on the VM) ------
DENSE_MODEL = "BAAI/bge-base-en-v1.5"   # 768-d dense, cosine
DENSE_DIM = 768

# Sparse retrievers. BM25 is maximally lexical -> the sharper divergence detector
# (the prior) AND the baseline's fusion sparse. miniCOIL is word-sense-aware and leans
# semantic -> kept indexed as the optional "production sparse upgrade" docs mention.
# Which one drives divergence is decided by validation AUC, not assumed here.
BM25_MODEL = "Qdrant/bm25"
MINICOIL_MODEL = "Qdrant/minicoil-v1"

# --- v2 architecture ---------------------------------------------------------
# The BASELINE is fusion-only (dense + BM25 -> RRF/DBSF), NO cross-encoder. This is
# the v2 simplification: v1 put bge-reranker IN the baseline, which inflated it
# ~5-12 nDCG (shrinking headroom) and SATURATED the max_score signal to AUC 0.53.
# In v2 the weakness signals are read off the FUSION scores instead, and the
# cross-encoder becomes a corrective ACTION + the cost-matched comparison baseline.
BASELINE_MODE = "hybrid"

# Fusion method for the hybrid baseline. RRF (rank-only, k=60) is the teachable
# default; DBSF (distribution-based: normalize each retriever by mean+-3sigma, then
# sum) preserves score magnitude, so it can be a sharper SIGNAL SUBSTRATE (height /
# spread have real dynamic range, unlike rank-compressed RRF). Both are server-side
# in Qdrant (no model, no extra latency). Which one is the better substrate is
# decided by validation signal-AUC and recorded in thresholds.json (`_fusion`);
# this constant is only the default before that selection. RRF is the teachable
# default; DBSF was measured head-to-head as the alternative substrate.
FUSION_METHOD = "rrf"   # "rrf" | "dbsf"

# Cross-encoder reranker = the corrective ACTION (NOT baseline machinery). jina v2
# is an upgrade over v1's bge-reranker-base and is in FastEmbed's registry
# (jina-v3 and bge-reranker-v2-m3 are NOT). Local ONNX, cached on the VM.
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# ColBERT late-interaction multivector (Tier 2: the Qdrant-multivector showcase +
# token-level near-miss recovery). FastEmbed LateInteraction; Qdrant MultiVectorConfig
# + MaxSim. answerai-colbert-small-v1 keeps the re-embed tractable on the VM.
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96   # per-token dim of answerai-colbert-small-v1 (MaxSim multivector)

# Named vector keys in the Qdrant collection.
DENSE_VEC = "dense"
BM25_VEC = "bm25"
MINICOIL_VEC = "minicoil"
COLBERT_VEC = "colbert"   # added in Tier 2 (re-embed); absent in the Tier-1 collection

# Role assignments for the sparse models (PRIORS; the divergence detector is
# confirmed by validation AUC, not assumed).
#   - Divergence detector: BM25, the maximally-lexical retriever, so its
#     disagreement with dense is the sharpest weakness signal.
#   - Hybrid baseline sparse: miniCOIL (Dylan's call: feature Qdrant's in-house,
#     word-sense-aware sparse model). MEASURED comparable-to-marginally-better than BM25
#     as the baseline on validation retrieval (+0.5pp full_gold, +1.25pp recall, within
#     noise), so the in-house value breaks the tie - chosen by measurement, not assumed.
#     BM25 stays indexed as a divergence-detector candidate (the detector is selected by
#     validation AUC) and the universal lexical reference.
DIVERGENCE_SPARSE_VEC = BM25_VEC
HYBRID_SPARSE_VEC = MINICOIL_VEC

# --- Retrieval constants -----------------------------------------------------
RETRIEVE_N = 50   # per-retriever raw top-N before fusion / cheap gate
TOP_K = 10        # pool / signal window (divergence, spread read over the top-10)

# v2.5 PRECISION REGIME: the agent answers from a FOCUSED top-k context (not all of
# TOP_K), so RANKING precision matters and the corrective tiers earn their keep. At
# top-10 single-hop retrieval is 98% complete (nothing to fix); at top-3 the gold is
# at rank 1 only ~77% of the time, so the cost-escalation ladder has real headroom on
# BOTH single-hop (precision: ColBERT) and multi-hop (recall: decompose). The headline
# metric is precision (recall@1/@3, MRR) by query type.
ANSWER_K = 3      # focused passages the LLM reads to answer
EVAL_KS = (1, 3)  # precision cutoffs reported in the headline

# --- LLM models (via LiteLLM; cross-provider judge reduces self-preference) ---
AGENT_MODEL = "anthropic/claude-sonnet-4-6"  # the agent (decompose + answer)
FAST_MODEL = "anthropic/claude-haiku-4-5"    # the fast sufficiency autorater (STOP)
JUDGE_MODEL = "openai/gpt-5.5"               # the answer judge (eval only)

# --- Loop budgets (so the loop always terminates) ----------------------------
# Ladder: baseline -> (RERANK) -> (IRCoT decompose) -> ANSWER/STOP.
DECOMPOSE_BUDGET = 1   # IRCoT decompose runs allowed per question
STEP_BUDGET = 4        # hard tool-call budget
IRCOT_MAX_HOPS = 4     # IRCoT: max retrieve+reason hops (MuSiQue is 2-4 hop)
MAX_SUBQUERIES = 4     # cap sub-queries accumulated across IRCoT hops
