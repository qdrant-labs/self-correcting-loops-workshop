"""Phase 2: embed the corpus and upsert it into Qdrant with named vectors.

One Qdrant collection, three vectors per point:
  - dense    : BAAI/bge-base-en-v1.5 (768-d, cosine)
  - bm25     : Qdrant/bm25            (sparse, IDF modifier)  - lexical / divergence
  - minicoil : Qdrant/minicoil-v1     (sparse, IDF modifier)  - production sparse

Payload carries title, text, and per-hop gold membership (supports), so a trace
inspector can see directly whether a missing hop's paragraph was retrieved.

Idempotent: drops and recreates the collection. On the VM this runs once at build
time; the room never re-indexes.

Usage:
  python scripts/setup_collections.py --limit 256   # quick end-to-end validation
  python scripts/setup_collections.py               # full corpus
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import config  # noqa: E402
import data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N docs (validation)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--upsert-batch", type=int, default=256)
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient, models
    from tqdm import tqdm

    corpus = data.load_corpus()
    doc_ids = sorted(corpus.keys())  # deterministic order
    if args.limit:
        doc_ids = doc_ids[: args.limit]
    docs = [corpus[d] for d in doc_ids]
    texts = [data.doc_embed_text(d) for d in docs]
    n = len(docs)
    print(f"corpus: {n} docs to index")

    print("loading FastEmbed models (first run downloads + caches) ...")
    dense_model = TextEmbedding(model_name=config.DENSE_MODEL)
    bm25_model = SparseTextEmbedding(model_name=config.BM25_MODEL)
    minicoil_model = SparseTextEmbedding(model_name=config.MINICOIL_MODEL)

    def embed_dense() -> list[list[float]]:
        t0 = time.time()
        out = [
            v.tolist()
            for v in tqdm(dense_model.embed(texts, batch_size=args.batch_size), total=n, desc="dense")
        ]
        print(f"  dense embedded in {time.time()-t0:.1f}s")
        return out

    def embed_sparse(model, label) -> list:
        t0 = time.time()
        out = list(tqdm(model.embed(texts, batch_size=args.batch_size), total=n, desc=label))
        print(f"  {label} embedded in {time.time()-t0:.1f}s")
        return out

    dense_vecs = embed_dense()
    bm25_vecs = embed_sparse(bm25_model, "bm25")
    minicoil_vecs = embed_sparse(minicoil_model, "minicoil")

    client = QdrantClient(url=config.QDRANT_URL, timeout=120)
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config={config.DENSE_VEC: models.VectorParams(size=config.DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            config.BM25_VEC: models.SparseVectorParams(modifier=models.Modifier.IDF),
            config.MINICOIL_VEC: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    print(f"created collection {config.COLLECTION!r}")

    def to_sparse(emb) -> "models.SparseVector":
        return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    t0 = time.time()
    for start in tqdm(range(0, n, args.upsert_batch), desc="upsert"):
        end = min(start + args.upsert_batch, n)
        points = []
        for i in range(start, end):
            doc = docs[i]
            points.append(
                models.PointStruct(
                    id=doc["doc_id"],
                    vector={
                        config.DENSE_VEC: dense_vecs[i],
                        config.BM25_VEC: to_sparse(bm25_vecs[i]),
                        config.MINICOIL_VEC: to_sparse(minicoil_vecs[i]),
                    },
                    payload={"title": doc["title"], "text": doc["text"], "supports": doc.get("supports", [])},
                )
            )
        client.upsert(collection_name=config.COLLECTION, points=points, wait=True)
    print(f"upserted {n} points in {time.time()-t0:.1f}s")

    # --- gate: count sane + gold reachable by id --------------------------------
    count = client.count(config.COLLECTION, exact=True).count
    print(f"\ncollection count = {count} (expected {n})")
    assert count == n, "collection count mismatch"

    if not args.limit:
        # every answerable question's gold must be retrievable by id
        questions = [q for q in data.load_questions() if q.get("answerable")]
        gold_ids = sorted({g for q in questions for g in q["gold_doc_ids"]})
        missing = []
        for start in range(0, len(gold_ids), 256):
            batch = gold_ids[start : start + 256]
            got = {p.id for p in client.retrieve(config.COLLECTION, ids=batch, with_payload=False, with_vectors=False)}
            missing.extend(g for g in batch if g not in got)
        print(f"gold docs checked: {len(gold_ids)} | missing by id: {len(missing)}")
        assert not missing, f"{len(missing)} gold docs not retrievable by id"
        print("gate: count sane OK; all gold retrievable by id OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
