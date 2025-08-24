
from typing import Dict, Any, List
from uuid import uuid5, NAMESPACE_DNS
from random import Random
import os, time
from qdrant_client import QdrantClient, models as qm
from tqdm import tqdm
from ..config import COLLECTION, BATCH_SIZE, MAX_DOCS, MAX_CHARS, EMB_MODEL
from ..models.embedder import DenseEmbedder
from ..logging import get_logger
logger=get_logger(__name__)
def _to_point_id(doc_id: Any):
    s=str(doc_id); return int(s) if s.isdigit() else str(uuid5(NAMESPACE_DNS, f"beir::{s}"))
def recreate_collection_dense(client: QdrantClient, dim: int):
    existing=[c.name for c in client.get_collections().collections]
    if COLLECTION in existing: client.delete_collection(collection_name=COLLECTION)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
        sparse_vectors_config=None,
    )
    logger.info("Collection %s ready (dense, dim=%d).", COLLECTION, dim)
def _prep(meta: Dict[str,str]) -> str:
    title=(meta.get("title") or "").strip(); body=(meta.get("text") or "").strip()
    txt=(title+" "+body).strip()
    if MAX_CHARS and MAX_CHARS>0 and len(txt)>MAX_CHARS: txt=txt[:MAX_CHARS]
    return txt
def index_corpus_dense(client: QdrantClient, corpus: Dict[str, Dict[str,str]]):
    items=list(corpus.items()); Random(42).shuffle(items)
    if MAX_DOCS and MAX_DOCS>0: items=items[:MAX_DOCS]
    emb=DenseEmbedder(EMB_MODEL); recreate_collection_dense(client, emb.dim)
    total=len(items); logger.info("Indexing %d documents (dense)...", total)
    for bi in tqdm(range(0,total,BATCH_SIZE)):
        chunk=items[bi:bi+BATCH_SIZE]
        ids=[doc_id for doc_id,_ in chunk]
        texts=[_prep(meta) for _,meta in chunk]
        t0=time.time(); vecs=emb.encode(texts); t1=time.time()
        pts=[qm.PointStruct(id=_to_point_id(ids[j]), vector={"dense": vecs[j].tolist()}, payload={"doc_id":str(ids[j]),"text":texts[j]}) for j in range(len(ids))]
        client.upsert(collection_name=COLLECTION, points=pts); t2=time.time()
        logger.info("[batch %d] embed: %.2fs | upsert: %.2fs | total: %.2fs", bi//BATCH_SIZE, (t1-t0), (t2-t1), (t2-t0))
    logger.info("Indexing finished.")
