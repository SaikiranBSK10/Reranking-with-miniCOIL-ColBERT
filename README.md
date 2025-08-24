
# Dense recall + MiniCOIL / ColBERT rerank (Qdrant)

Two-stage retrieval:
1) Dense recall with MiniLM → Qdrant
2) Rerank with MiniCOIL **or** ColBERT

## Quickstart
```bash
docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m dense_rerank_demo.scripts.ingest
python -m dense_rerank_demo.scripts.query --q "Vitamin D supplementation reduces respiratory infections." --reranker minicoil --k 100 --show 10
python -m dense_rerank_demo.scripts.eval_beir --limit 50 --reranker minicoil --k 100 --covered-only
```
Set knobs in `.env` (optional): `MAX_DOCS`, `TOPK_RECALL`, `COLBERT_CKPT`, etc.
