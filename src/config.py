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

# --- Embedding models (FastEmbed, local ONNX, cached on the VM) ---------------
DENSE_MODEL = "BAAI/bge-base-en-v1.5"   # 768-d dense, cosine
DENSE_DIM = 768

# Sparse retriever: miniCOIL, Qdrant's word-sense-aware sparse model. It is the
# baseline's fusion sparse and the sparse ranking the divergence signal reads.
MINICOIL_MODEL = "Qdrant/minicoil-v1"

# --- v2 architecture ---------------------------------------------------------
# The BASELINE is fusion-only (dense + miniCOIL -> RRF), NO cross-encoder. Putting a
# reranker IN the baseline inflates it and SATURATES the max_score signal; instead the
# weakness signals are read off the FUSION scores, and ColBERT late interaction (Tier 2)
# is the corrective ACTION.
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

# ColBERT late-interaction multivector (Tier 2: the Qdrant-multivector showcase +
# token-level near-miss recovery). FastEmbed LateInteraction; Qdrant MultiVectorConfig
# + MaxSim. answerai-colbert-small-v1 keeps the re-embed tractable on the VM.
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96   # per-token dim of answerai-colbert-small-v1 (MaxSim multivector)

# Named vector keys in the Qdrant collection.
DENSE_VEC = "dense"
MINICOIL_VEC = "minicoil"
COLBERT_VEC = "colbert"   # added in Tier 2 (re-embed); absent in the Tier-1 collection

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

# --- LLM models (via LiteLLM) ------------------------------------------------
AGENT_MODEL = "anthropic/claude-sonnet-4-6"  # the agent (decompose + answer)
FAST_MODEL = "anthropic/claude-haiku-4-5"    # the fast sufficiency autorater (STOP)

# --- Loop budgets (so the loop always terminates) ----------------------------
# Ladder: baseline -> (ColBERT) -> (IRCoT decompose) -> ANSWER/STOP.
DECOMPOSE_BUDGET = 1   # IRCoT decompose runs allowed per question
STEP_BUDGET = 4        # hard tool-call budget
IRCOT_MAX_HOPS = 4     # IRCoT: max retrieve+reason hops (MuSiQue is 2-4 hop)
MAX_SUBQUERIES = 4     # cap sub-queries accumulated across IRCoT hops
