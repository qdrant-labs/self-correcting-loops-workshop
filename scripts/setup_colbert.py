"""Tier 2: build the `musique_colbert` collection (dense + ColBERT multivector).

A SEPARATE collection from the Tier-1 `musique` (which stays untouched and validated):
  - dense   : BAAI/bge-base-en-v1.5 (768-d, cosine) - reused from `musique` (retrieved by
              id, NOT re-embedded), so it is byte-identical and the dense prefetch matches.
  - colbert : answerdotai/answerai-colbert-small-v1 (96-d per token; Qdrant MultiVectorConfig
              + MaxSim comparator) - the late-interaction multivector showcased in Tier 2.

The ColBERT deep-retrieval action (retrieval.colbert_search) prefetches with dense, then
rescores the candidates with ColBERT MaxSim. This script only needs dense + colbert.

Usage:
  python scripts/setup_colbert.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import config  # noqa: E402
import data  # noqa: E402

warnings.filterwarnings("ignore")
from fastembed import LateInteractionTextEmbedding, TextEmbedding  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402
from tqdm import tqdm  # noqa: E402

EMBED_BATCH = 64   # docs per ColBERT forward pass
UPSERT_BATCH = 128  # points per upsert request


def main() -> int:
    corpus = data.load_corpus()
    doc_ids = sorted(corpus.keys())  # SAME deterministic order as setup_collections.py
    docs = [corpus[d] for d in doc_ids]
    texts = [data.doc_embed_text(d) for d in docs]
    n = len(docs)
    print(f"corpus: {n} docs to index into {config.COLBERT_COLLECTION!r}")

    client = QdrantClient(url=config.QDRANT_URL, timeout=300)
    if not client.collection_exists(config.COLLECTION):
        raise SystemExit(f"ERROR: Tier-1 collection {config.COLLECTION!r} missing; build it first "
                         "(scripts/setup_collections.py) - dense vectors are reused from it.")

    print("loading ColBERT (answerai-colbert-small-v1; first run downloads + caches) ...")
    colbert_model = LateInteractionTextEmbedding(model_name=config.COLBERT_MODEL)
    t0 = time.time()
    colbert_vecs = [
        [row.tolist() for row in v]  # (n_tokens, 96) -> list[list[float]]
        for v in tqdm(colbert_model.embed(texts, batch_size=EMBED_BATCH), total=n, desc="colbert")
    ]
    print(f"  colbert embedded in {time.time()-t0:.1f}s")

    if client.collection_exists(config.COLBERT_COLLECTION):
        client.delete_collection(config.COLBERT_COLLECTION)
    client.create_collection(
        collection_name=config.COLBERT_COLLECTION,
        vectors_config={
            config.DENSE_VEC: models.VectorParams(size=config.DENSE_DIM, distance=models.Distance.COSINE),
            config.COLBERT_VEC: models.VectorParams(
                size=config.COLBERT_DIM,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                hnsw_config=models.HnswConfigDiff(m=0),  # no HNSW on the multivector: it is a rescorer, reached via dense prefetch
            ),
        },
    )
    print(f"created collection {config.COLBERT_COLLECTION!r} (dense + colbert MaxSim multivector)")

    t0 = time.time()
    for start in tqdm(range(0, n, UPSERT_BATCH), desc="upsert"):
        end = min(start + UPSERT_BATCH, n)
        batch_ids = [docs[i]["doc_id"] for i in range(start, end)]
        # reuse dense from `musique` (byte-identical; no re-embed) - retrieve by id
        got = client.retrieve(config.COLLECTION, ids=batch_ids, with_payload=False, with_vectors=[config.DENSE_VEC])
        dense_by_id = {p.id: p.vector[config.DENSE_VEC] for p in got}
        points = []
        for i in range(start, end):
            doc = docs[i]
            dense = dense_by_id.get(doc["doc_id"])
            if dense is None:
                raise SystemExit(f"ERROR: dense vector missing in {config.COLLECTION!r} for {doc['doc_id']}")
            points.append(
                models.PointStruct(
                    id=doc["doc_id"],
                    vector={config.DENSE_VEC: dense, config.COLBERT_VEC: colbert_vecs[i]},
                    payload={"title": doc["title"], "text": doc["text"], "supports": doc.get("supports", [])},
                )
            )
        client.upsert(collection_name=config.COLBERT_COLLECTION, points=points, wait=True)
    print(f"upserted {n} points in {time.time()-t0:.1f}s")

    count = client.count(config.COLBERT_COLLECTION, exact=True).count
    print(f"\ncollection count = {count} (expected {n})")
    assert count == n, "collection count mismatch"

    # smoke: a dense-prefetch -> colbert MaxSim query returns results
    q = "Who owns the Gold Spike in Las Vegas?"
    dense_q = next(iter(TextEmbedding(config.DENSE_MODEL).query_embed(q))).tolist()
    colbert_q = [r.tolist() for r in next(iter(colbert_model.query_embed(q)))]
    res = client.query_points(
        config.COLBERT_COLLECTION,
        prefetch=[models.Prefetch(query=dense_q, using=config.DENSE_VEC, limit=50)],
        query=colbert_q, using=config.COLBERT_VEC, limit=5, with_payload=True,
    ).points
    print(f"smoke query -> {len(res)} hits; top: {res[0].payload.get('title','')[:60]!r} (MaxSim {res[0].score:.3f})")
    print("gate: count sane OK; dense-prefetch + ColBERT MaxSim rescore OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
