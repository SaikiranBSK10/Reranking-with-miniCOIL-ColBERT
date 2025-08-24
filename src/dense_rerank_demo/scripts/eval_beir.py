import argparse, time, math
import numpy as np
from tqdm import tqdm

from ..config import (
    DATASET, DATA_DIR, EVAL_LIMIT, TOPK_SHOW, TOPK_RECALL, COLLECTION,
    SPARSE_MODEL, COLBERT_CKPT
)
from ..data.loader import load_beir
from ..qdrant.client import get_client
from ..qdrant.search import retrieve_dense
from ..models.embedder import Embedder
from ..models.reranker_minicoil import MiniCOILReranker
from ..models.reranker_colbert import ColbertReranker
from qdrant_client import QdrantClient

def dcg(scores):
    return sum((s / math.log2(i + 2)) for i, s in enumerate(scores))

def ndcg_at_10(binary):
    gains = binary[:10]
    ideal = sorted(binary, reverse=True)[:10]
    denom = dcg(ideal) or 1.0
    return dcg(gains) / denom

def mrr_at_10(binary):
    for i, rel in enumerate(binary[:10], 1):
        if rel:
            return 1.0 / i
    return 0.0

def precision_at_k(binary_labels, k=10) -> float:
    denom = min(k, len(binary_labels))
    return (sum(binary_labels[:k]) / denom) if denom else 0.0

def _indexed_ids(client: QdrantClient, collection: str) -> set:
    ids = set(); offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=collection, limit=1000, with_payload=["doc_id"], offset=offset
        )
        for p in pts:
            ids.add(p.payload["doc_id"])
        if offset is None:
            break
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=EVAL_LIMIT)
    ap.add_argument("--k", type=int, default=TOPK_RECALL)
    ap.add_argument("--reranker", choices=["minicoil", "colbert"], default="minicoil")
    ap.add_argument("--covered-only", action="store_true")
    args = ap.parse_args()

    corpus, queries, qrels = load_beir(DATASET, DATA_DIR, split="test")
    client = get_client()

    qids = list(queries.keys())
    if args.covered_only:
        indexed = _indexed_ids(client, COLLECTION)
        qids = [qid for qid in qids if any(d in indexed for d in qrels.get(qid, {}))]
    qids = qids[:args.limit]

    # You can embed here and pass vectors, or just pass text and let another
    # function embed. Here we embed explicitly to match your API.
    emb = Embedder()
    reranker = MiniCOILReranker(SPARSE_MODEL) if args.reranker == "minicoil" else ColbertReranker(COLBERT_CKPT)

    ndcg_pre, ndcg_post = [], []
    mrr_pre,  mrr_post  = [], []
    p10_pre,  p10_post  = [], []
    t_rec, t_rr = [], []

    for qid in tqdm(qids):
        q = queries[qid]

        # Recall (dense): embed then search the named vector "dense"
        t0 = time.time()
        qvec = emb.encode([q])[0]
        cands = retrieve_dense(client, qvec, topk=args.k)
        t1 = time.time()

        # Rerank
        post = reranker.rerank(q, cands)[:TOPK_SHOW]
        t2 = time.time()

        t_rec.append(t1 - t0); t_rr.append(t2 - t1)

        pre = cands[:TOPK_SHOW]
        rel = qrels.get(qid, {})  # {doc_id: grade}

        tobin = lambda did: 1 if rel.get(did, 0) > 0 else 0
        y_pre  = [tobin(d["id"]) for d in pre]
        y_post = [tobin(d["id"]) for d in post]

        ndcg_pre.append(ndcg_at_10(y_pre)); ndcg_post.append(ndcg_at_10(y_post))
        mrr_pre.append(mrr_at_10(y_pre));   mrr_post.append(mrr_at_10(y_post))
        p10_pre.append(precision_at_k(y_pre, k=10)); p10_post.append(precision_at_k(y_post, k=10))

    if not qids:
        print("No queries to evaluate (check --covered-only or MAX_DOCS).")
        return

    print(f"\n=== Evaluation (avg over {len(qids)} queries) ===")
    print(f"nDCG@10  no-rerank: {np.mean(ndcg_pre):.4f}  | rerank: {np.mean(ndcg_post):.4f}")
    print(f"MRR@10   no-rerank: {np.mean(mrr_pre):.4f}   | rerank: {np.mean(mrr_post):.4f}")
    print(f"P@10     no-rerank: {np.mean(p10_pre):.4f}   | rerank: {np.mean(p10_post):.4f}")

    rec_ms = 1000*np.array(t_rec); rr_ms = 1000*np.array(t_rr)
    print(f"Latency p50 — recall: {np.percentile(rec_ms,50):.1f} ms | "
          f"rerank: {np.percentile(rr_ms,50):.1f} ms | "
          f"total: {np.percentile(rec_ms+rr_ms,50):.1f} ms")

if __name__ == "__main__":
    main()
