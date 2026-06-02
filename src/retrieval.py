"""The retrieval pipeline.

    query
      -> encode dense + bm25 + minicoil (query side)
      -> raw dense / bm25 / minicoil top-N rankings   [cheap; feeds divergence]
      -> fuse dense + BM25 per the mode (Qdrant Query API: Prefetch + RRF/DBSF)
      -> candidates carry the FUSION score (the v2 signal substrate)

The v2 BASELINE is fusion-only: NO cross-encoder in the default path. The weakness
signals (height / spread) are read off the FUSION scores (signals.py), which kills
the v1 reranker saturation. The cross-encoder (jina-reranker-v2) is a corrective
ACTION applied to the fused pool, and the substrate of the cost-matched comparison
baseline (expand-k + rerank).

Modes (single-query):
  vector  - dense only (the weak default; score = cosine similarity).
  hybrid  - dense fused with BM25 via RRF/DBSF (the baseline; score = fusion score).

Actions (applied to a RetrievalResult / pool, not baseline machinery):
  rerank(result)            - cross-encode the fused pool with jina-v2 -> top-K.

IRCoT (iterative decompose) is LLM-driven and lives in agent.py; it calls `search`
per hop and `union_pool` to merge the per-hop evidence.

`search` returns the top-K answer set in `candidates` AND a deeper `pool` (up to n
fused candidates) so rerank/expand-k have material to reorder. Per-stage latency is
recorded in `timings_ms` (signals are cheap relative to an LLM judge, not free).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import config
import data

# --- lazy singletons ---------------------------------------------------------
_models: dict = {}
_client = None
_colbert = None


def get_models() -> dict:
    if not _models:
        from fastembed import SparseTextEmbedding, TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _models["dense"] = TextEmbedding(config.DENSE_MODEL)
        _models["bm25"] = SparseTextEmbedding(config.BM25_MODEL)
        _models["minicoil"] = SparseTextEmbedding(config.MINICOIL_MODEL)
        _models["reranker"] = TextCrossEncoder(config.RERANKER_MODEL)
    return _models


def get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=config.QDRANT_URL, timeout=120)
    return _client


# --- result types ------------------------------------------------------------
@dataclass
class Candidate:
    doc_id: str
    title: str
    text: str
    score: float                                   # the ACTIVE score (fusion or rerank)
    supports: list = field(default_factory=list)   # [{question_id, hop_index}]


@dataclass
class RetrievalResult:
    query: str
    mode: str
    sub_queries: list[str]
    candidates: list[Candidate]                  # the top-K answer set
    raw: dict[str, list[tuple[str, float]]]      # {"dense"/"bm25"/"minicoil": [(id, score), ...]} top-N
    timings_ms: dict[str, float]
    score_kind: str = "fusion"                   # "fusion" | "rerank" | "dense"
    pool: list[Candidate] = field(default_factory=list)  # deeper fused pool (for rerank / expand-k)

    @property
    def doc_ids(self) -> list[str]:
        return [c.doc_id for c in self.candidates]

    @property
    def scores(self) -> list[float]:
        return [c.score for c in self.candidates]

    @property
    def latency_ms(self) -> float:
        return sum(self.timings_ms.values())


# --- query encoding ----------------------------------------------------------
def encode_query(query: str):
    """Returns (dense_vector, bm25_sparse_embedding, minicoil_sparse_embedding).
    Uses query_embed so bge gets its query instruction and the sparse models get
    query-side weighting; IDF is applied server-side by Qdrant."""
    m = get_models()
    dense = next(iter(m["dense"].query_embed(query))).tolist()
    bm25 = next(iter(m["bm25"].query_embed(query)))
    minicoil = next(iter(m["minicoil"].query_embed(query)))
    return dense, bm25, minicoil


def _sparse(emb):
    from qdrant_client import models

    return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def _sigmoid(z: float) -> float:
    # stable logistic: cross-encoder logits -> relevance probability in (0, 1), so a
    # reranked score is an interpretable probability. Monotonic -> ranking unchanged.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _fusion_query(fusion: str | None):
    from qdrant_client import models

    f = (fusion or config.FUSION_METHOD).lower()
    return models.FusionQuery(fusion=models.Fusion.DBSF if f == "dbsf" else models.Fusion.RRF)


# --- raw rankings (divergence inputs) ----------------------------------------
def raw_rankings(query: str, n: int = config.RETRIEVE_N, encoded=None) -> dict[str, list[tuple[str, float]]]:
    """Three independent top-N rankings (dense, bm25, minicoil). No fusion - the
    cheap substrate the divergence signal reads (dense vs bm25)."""
    c = get_client()
    dense, bm25, minicoil = encoded or encode_query(query)
    out: dict[str, list[tuple[str, float]]] = {}
    r = c.query_points(config.COLLECTION, query=dense, using=config.DENSE_VEC, limit=n, with_payload=False).points
    out["dense"] = [(p.id, p.score) for p in r]
    for name, emb, vec in (("bm25", bm25, config.BM25_VEC), ("minicoil", minicoil, config.MINICOIL_VEC)):
        r = c.query_points(config.COLLECTION, query=_sparse(emb), using=vec, limit=n, with_payload=False).points
        out[name] = [(p.id, p.score) for p in r]
    return out


# --- fused points per mode ---------------------------------------------------
def _fused_points(query: str, mode: str, n: int, fusion: str | None, encoded=None):
    from qdrant_client import models

    c = get_client()
    dense, bm25, minicoil = encoded or encode_query(query)
    if mode == "vector":
        return c.query_points(config.COLLECTION, query=dense, using=config.DENSE_VEC, limit=n, with_payload=True).points
    if mode == "hybrid":
        sparse_vec = config.HYBRID_SPARSE_VEC
        sparse_emb = minicoil if sparse_vec == config.MINICOIL_VEC else bm25
        prefetch = [
            models.Prefetch(query=dense, using=config.DENSE_VEC, limit=n),
            models.Prefetch(query=_sparse(sparse_emb), using=sparse_vec, limit=n),
        ]
        return c.query_points(
            config.COLLECTION,
            prefetch=prefetch,
            query=_fusion_query(fusion),
            limit=n,
            with_payload=True,
        ).points
    raise ValueError(f"unknown single-query mode: {mode!r}")


def _candidates_from_points(points) -> list[Candidate]:
    return [
        Candidate(
            doc_id=p.id,
            title=p.payload.get("title", ""),
            text=p.payload.get("text", ""),
            score=float(p.score),
            supports=p.payload.get("supports", []),
        )
        for p in points
    ]


# --- entry point: the baseline retrieval (no rerank) -------------------------
def search(
    query: str,
    mode: str = config.BASELINE_MODE,
    n: int = config.RETRIEVE_N,
    k: int = config.TOP_K,
    fusion: str | None = None,
    encoded=None,
    with_raw: bool = True,
) -> RetrievalResult:
    """Single-query retrieval for `vector` or `hybrid`. Returns the top-K answer set
    (candidates carry the FUSION score) plus the deeper fused `pool` so the rerank /
    expand-k actions have material. NO cross-encoder here - that is a separate action."""
    timings: dict[str, float] = {}
    t = time.perf_counter()
    enc = encoded or encode_query(query)
    timings["encode_ms"] = (time.perf_counter() - t) * 1000

    raw = {}
    if with_raw:
        t = time.perf_counter()
        raw = raw_rankings(query, n, encoded=enc)
        timings["raw_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    points = _fused_points(query, mode, n, fusion, encoded=enc)
    timings["fuse_ms"] = (time.perf_counter() - t) * 1000

    pool = _candidates_from_points(points)
    score_kind = "dense" if mode == "vector" else "fusion"
    return RetrievalResult(query, mode, [], pool[:k], raw, timings,
                           score_kind=score_kind, pool=pool)


# --- action: cross-encoder rerank of a fused pool ----------------------------
def _rerank_candidates(query: str, pool: list[Candidate], k: int) -> list[Candidate]:
    if not pool:
        return []
    m = get_models()
    docs = [data.doc_embed_text({"title": c.title, "text": c.text}) for c in pool]
    scores = [_sigmoid(float(z)) for z in m["reranker"].rerank(query, docs)]
    ranked = sorted(zip(pool, scores), key=lambda cs: cs[1], reverse=True)[:k]
    return [Candidate(c.doc_id, c.title, c.text, float(s), c.supports) for c, s in ranked]


def rerank(result: RetrievalResult, query: str | None = None, k: int = config.TOP_K) -> RetrievalResult:
    """Corrective ACTION: cross-encode the fused pool with jina-v2 and keep the top-K.
    Reorders the existing candidate pool (a precision fix); it does not retrieve more.
    Returns a new RetrievalResult with score_kind='rerank' (scores are relevance
    probabilities in (0,1)). Carries the original raw rankings (divergence unchanged)."""
    q = query or result.query
    pool = result.pool or result.candidates
    t = time.perf_counter()
    cands = _rerank_candidates(q, pool, k)
    timings = dict(result.timings_ms)
    timings["rerank_ms"] = (time.perf_counter() - t) * 1000
    return RetrievalResult(result.query, result.mode, list(result.sub_queries), cands,
                           result.raw, timings, score_kind="rerank", pool=pool)


# --- Tier 2: ColBERT late-interaction deep retrieval -------------------------
def get_colbert_model():
    """Lazy, SEPARATE from get_models so Tier-1 retrieval never loads ColBERT."""
    global _colbert
    if _colbert is None:
        from fastembed import LateInteractionTextEmbedding

        _colbert = LateInteractionTextEmbedding(config.COLBERT_MODEL)
    return _colbert


def colbert_search(query: str, n_prefetch: int = config.RETRIEVE_N, k: int = config.TOP_K,
                   dense_vec=None) -> RetrievalResult:
    """Tier-2 ColBERT deep-retrieval ACTION: prefetch a deep dense pool from the
    `musique_colbert` collection, then rescore it with ColBERT late-interaction MaxSim
    -> top-K. The late-interaction peer of the cross-encoder rerank action (MaxSim vs a
    cross-encoder as the rescorer). score_kind='colbert' (MaxSim score)."""
    from qdrant_client import models

    c = get_client()
    dv = dense_vec if dense_vec is not None else encode_query(query)[0]
    t = time.perf_counter()
    cq = [r.tolist() for r in next(iter(get_colbert_model().query_embed(query)))]
    points = c.query_points(
        config.COLBERT_COLLECTION,
        prefetch=[models.Prefetch(query=dv, using=config.DENSE_VEC, limit=n_prefetch)],
        query=cq, using=config.COLBERT_VEC, limit=k, with_payload=True,
    ).points
    timings = {"colbert_ms": (time.perf_counter() - t) * 1000}
    return RetrievalResult(query, "colbert", [], _candidates_from_points(points), {}, timings,
                           score_kind="colbert")


# --- IRCoT pooling helper (the driver itself is LLM-driven, in agent.py) -----
def union_pool(results: list[RetrievalResult], k: int = config.TOP_K) -> list[Candidate]:
    """Union the candidate pools of several per-hop retrievals, keeping the MAX score
    per doc, and take the top-K. Used by the IRCoT driver to merge per-hop evidence.
    Pools the fused `candidates` (top-K of each hop) so each hop's strongest passages
    can surface into the final set - the mechanism that recovers a missing hop."""
    pooled: dict[str, Candidate] = {}
    for res in results:
        for c in res.candidates:
            cur = pooled.get(c.doc_id)
            if cur is None or c.score > cur.score:
                pooled[c.doc_id] = c
    return sorted(pooled.values(), key=lambda c: c.score, reverse=True)[:k]
