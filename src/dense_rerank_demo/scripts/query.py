
import argparse, time
from ..qdrant.client import get_client
from ..qdrant.search import retrieve_dense
from ..models.reranker_minicoil import MiniCOILReranker
from ..models.reranker_colbert import ColbertReranker
from ..config import TOPK_SHOW, COLBERT_CKPT, SPARSE_MODEL
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--show", type=int, default=TOPK_SHOW)
    ap.add_argument("--reranker", choices=["minicoil","colbert","none"], default="minicoil")
    args=ap.parse_args()
    client=get_client()
    t0=time.time(); cands=retrieve_dense(client, args.q, k=args.k); t1=time.time()
    if args.reranker=="none":
        post=cands[:args.show]
    elif args.reranker=="minicoil":
        rr=MiniCOILReranker(SPARSE_MODEL); post=rr.rerank(args.q, cands)[:args.show]
    else:
        rr=ColbertReranker(COLBERT_CKPT); post=rr.rerank(args.q, cands)[:args.show]
    t2=time.time()
    print("\n=== Before (dense-only) ===")
    for i,c in enumerate(cands[:args.show],1): print(f"{i:2d}. ({c['score']:.3f}) {c['text'][:180]}...")
    print(f"\n=== After (reranked: {args.reranker}) ===")
    pre_rank={c['id']:i+1 for i,c in enumerate(cands[:args.show])}
    for i,c in enumerate(post,1):
        was=pre_rank.get(c['id']); tag=f" (was #{was})" if was else ""
        rs=c.get('rerank_score',0.0)
        print(f"{i:2d}. ({rs:.3f}){tag} {c['text'][:180]}...")
    print(f"\nLatency — recall: {(t1-t0)*1000:.1f} ms | rerank: {(t2-t1)*1000:.1f} ms | total: {(t2-t0)*1000:.1f} ms")
if __name__=="__main__": main()
