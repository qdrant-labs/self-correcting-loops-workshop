"""Phase 2: embed the corpus and upsert it into Qdrant with named vectors.

One Qdrant collection, two vectors per point:
  - dense    : BAAI/bge-base-en-v1.5 (768-d, cosine)
  - minicoil : Qdrant/minicoil-v1     (sparse, IDF modifier)  - word-sense-aware sparse

Payload carries title, text, and per-hop gold membership (supports), so a trace
inspector can see directly whether a missing hop's paragraph was retrieved.

Idempotent: drops and recreates the collection. On the VM this runs once at build
time; the room never re-indexes.

Usage:
  python scripts/setup_collections.py
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
from fastembed import SparseTextEmbedding, TextEmbedding  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402
from tqdm import tqdm  # noqa: E402

EMBED_BATCH = 128   # docs per embedding forward pass
UPSERT_BATCH = 256  # points per upsert request


def main() -> int:
    corpus = data.load_corpus()
    doc_ids = sorted(corpus.keys())  # deterministic order
    docs = [corpus[d] for d in doc_ids]
    texts = [data.doc_embed_text(d) for d in docs]
    n = len(docs)
    print(f"corpus: {n} docs to index")

    print("loading FastEmbed models (first run downloads + caches) ...")
    dense_model = TextEmbedding(model_name=config.DENSE_MODEL)
    minicoil_model = SparseTextEmbedding(model_name=config.MINICOIL_MODEL)

    def embed(model, label):
        t0 = time.time()
        out = list(tqdm(model.embed(texts, batch_size=EMBED_BATCH), total=n, desc=label))
        print(f"  {label} embedded in {time.time()-t0:.1f}s")
        return out

    dense_vecs = [v.tolist() for v in embed(dense_model, "dense")]
    minicoil_vecs = embed(minicoil_model, "minicoil")

    client = QdrantClient(url=config.QDRANT_URL, timeout=120)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config={
            config.DENSE_VEC: models.VectorParams(size=config.DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            config.MINICOIL_VEC: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    print(f"created collection {config.COLLECTION!r}")

    t0 = time.time()
    for start in tqdm(range(0, n, UPSERT_BATCH), desc="upsert"):
        points = []
        for i in range(start, min(start + UPSERT_BATCH, n)):
            doc = docs[i]
            minicoil = minicoil_vecs[i]
            points.append(
                models.PointStruct(
                    id=doc["doc_id"],
                    vector={
                        config.DENSE_VEC: dense_vecs[i],
                        config.MINICOIL_VEC: models.SparseVector(
                            indices=minicoil.indices.tolist(), values=minicoil.values.tolist()
                        ),
                    },
                    payload={"title": doc["title"], "text": doc["text"], "supports": doc.get("supports", [])},
                )
            )
        client.upsert(collection_name=config.COLLECTION, points=points, wait=True)
    print(f"upserted {n} points in {time.time()-t0:.1f}s")

    # --- gate: count sane + every answerable question's gold reachable by id ----
    count = client.count(config.COLLECTION, exact=True).count
    print(f"\ncollection count = {count} (expected {n})")
    assert count == n, "collection count mismatch"

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
