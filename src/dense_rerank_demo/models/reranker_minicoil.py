
from typing import List, Dict, Any, Tuple
from functools import lru_cache
from fastembed import SparseTextEmbedding
class MiniCOILReranker:
    def __init__(self, model_name:str="Qdrant/minicoil-v1"):
        self.model=SparseTextEmbedding(model_name=model_name)
    @staticmethod
    def _dot(a_idx,a_val,b_idx,b_val):
        i=j=0; acc=0.0
        while i<len(a_idx) and j<len(b_idx):
            if a_idx[i]==b_idx[j]:
                acc+=a_val[i]*b_val[j]; i+=1; j+=1
            elif a_idx[i]<b_idx[j]: i+=1
            else: j+=1
        return acc
    @lru_cache(maxsize=10000)
    def _doc_cached(self, text:str) -> Tuple[Tuple[int,...],Tuple[float,...]]:
        sv=next(self.model.embed([text]))
        return tuple(sv.indices), tuple(sv.values)
    def rerank(self, query:str, candidates: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        q_sv=next(self.model.embed([query])); q_idx,q_val=q_sv.indices,q_sv.values
        scores=[]
        for c in candidates:
            d_idx,d_val=self._doc_cached(c["text"])
            scores.append(self._dot(q_idx,q_val,list(d_idx),list(d_val)))
        for s,c in zip(scores,candidates): c["rerank_score"]=float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
