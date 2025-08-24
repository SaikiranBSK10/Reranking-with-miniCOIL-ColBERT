from typing import List, Dict, Any
from qdrant_client import models as qm
from ..config import COLLECTION, TOPK_RECALL

def retrieve_dense(client, qvec: List[float], topk: int = TOPK_RECALL) -> List[Dict[str, Any]]:
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=qm.NamedVector(name="dense", vector=qvec),
        with_payload=True,
        limit=topk,
    )
    return [
        {
            "id": h.payload["doc_id"],
            "text": h.payload["text"],
            "score": float(h.score),
        }
        for h in hits
    ]
