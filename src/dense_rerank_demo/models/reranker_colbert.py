
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
from ..logging import get_logger
logger=get_logger(__name__)
@torch.inference_mode()
def _maxsim(q_emb, d_emb):
    sim = torch.matmul(q_emb, d_emb.transpose(0,1))
    return sim.max(dim=1).values.sum().item()
class ColbertReranker:
    def __init__(self, checkpoint: str = "colbert-ir/colbertv2.0"):
        logger.info("Loading HF ColBERT checkpoint for reranking: %s", checkpoint)
        self.tok=AutoTokenizer.from_pretrained(checkpoint)
        self.model=AutoModel.from_pretrained(checkpoint).eval()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cls_id=getattr(self.tok,"cls_token_id",None)
        self.sep_id=getattr(self.tok,"sep_token_id",None)
        self.has_linear=hasattr(self.model,"linear")
        self.max_q_len=64; self.max_d_len=300
    @torch.inference_mode()
    def _enc_tokens(self, texts, max_len):
        enc=self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        out=self.model(**enc)
        hs = out.last_hidden_state if hasattr(out,"last_hidden_state") else out[0]
        if self.has_linear: hs=self.model.linear(hs)
        hs=torch.nn.functional.normalize(hs,p=2,dim=-1)
        attn=enc["attention_mask"].bool()
        if self.cls_id is not None: attn = attn & (enc["input_ids"] != self.cls_id)
        if self.sep_id is not None: attn = attn & (enc["input_ids"] != self.sep_id)
        embs=[]; B,L,D=hs.shape
        for b in range(B):
            m=attn[b]; embs.append(hs[b][m])
        return embs
    def _enc_q(self,q): return self._enc_tokens([q], self.max_q_len)[0].cpu()
    def _enc_ds(self, texts):
        bs = 16 if self.device.type=="cuda" else 8
        out=[]; 
        for i in range(0,len(texts),bs):
            out+=self._enc_tokens(texts[i:i+bs], self.max_d_len)
        return [x.cpu() for x in out]
    def rerank(self, query:str, candidates: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        q_emb=self._enc_q(query); d_embs=self._enc_ds([c["text"] for c in candidates])
        scores=[(_maxsim(q_emb,d) if q_emb.numel() and d.numel() else float("-inf")) for d in d_embs]
        for s,c in zip(scores,candidates): c["rerank_score"]=float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
