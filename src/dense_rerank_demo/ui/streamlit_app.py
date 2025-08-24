# src/dense_rerank_demo/ui/streamlit_app.py
import time
import streamlit as st

st.set_page_config(page_title="Dense → MiniCOIL / ColBERT Rerank (Qdrant)", layout="wide")
st.title("Dense recall → MiniCOIL / ColBERT rerank (Qdrant)")

@st.cache_resource(show_spinner=False)
def _load_services():
    """Import and construct heavy stuff lazily so import errors surface in the UI."""
    try:
        # All project imports go INSIDE this function so errors are caught and shown.
        from dense_rerank_demo.config import (
            TOPK_RECALL, TOPK_SHOW, EMB_MODEL, SPARSE_MODEL, COLBERT_CKPT, COLLECTION
        )
        from dense_rerank_demo.qdrant.client import get_client
        from dense_rerank_demo.models.embedder import Embedder
        from dense_rerank_demo.qdrant.search import retrieve_dense
        from dense_rerank_demo.models.reranker_minicoil import MiniCOILReranker
        from dense_rerank_demo.models.reranker_colbert import ColbertReranker

        client = get_client()  # make sure prefer_grpc=False in client.py on your Mac
        emb = Embedder(EMB_MODEL)
        rr_minicoil = MiniCOILReranker(SPARSE_MODEL)
        rr_colbert = ColbertReranker(COLBERT_CKPT)

        return {
            "cfg": dict(TOPK_RECALL=TOPK_RECALL, TOPK_SHOW=TOPK_SHOW, COLLECTION=COLLECTION,
                        EMB_MODEL=EMB_MODEL, SPARSE_MODEL=SPARSE_MODEL, COLBERT_CKPT=COLBERT_CKPT),
            "client": client,
            "embedder": emb,
            "retrieve_dense": retrieve_dense,
            "rr_minicoil": rr_minicoil,
            "rr_colbert": rr_colbert,
        }
    except Exception as e:
        # Surface import/initialization errors in the UI instead of a blank page.
        st.error("Failed to initialize services. See details below.")
        st.exception(e)
        st.stop()

svc = _load_services()  # -> dict with client, embedder, retrieve_dense, rerankers, cfg

with st.sidebar:
    st.header("Settings")
    reranker = st.selectbox("Reranker", ["None", "MiniCOIL", "ColBERT"])
    k = st.slider("Recall K (candidates)", 10, 300, int(svc["cfg"]["TOPK_RECALL"]), 10)
    show_n = st.slider("Show top-N", 5, 20, int(svc["cfg"]["TOPK_SHOW"]), 1)
    show_text = st.checkbox("Show full text", False)
    st.divider()
    st.caption(f"Collection: `{svc['cfg']['COLLECTION']}`")
    st.caption(f"Embedder: `{svc['cfg']['EMB_MODEL']}`")
    st.caption("Tip: For ColBERT on CPU, keep K ≤ 50.")
    if st.button("Health check"):
        try:
            # collection exists?
            coll = svc["client"].get_collection(svc["cfg"]["COLLECTION"])
            # count points
            count = svc["client"].count(svc["cfg"]["COLLECTION"], exact=True).count
            st.success(f"Qdrant OK · vectors: {list(coll.config.params.vectors.items())[0][0]} · points: {count}")
        except Exception as e:
            st.error("Qdrant health check failed:")
            st.exception(e)

q = st.text_input("Ask a question:", placeholder="e.g., Does vitamin D reduce respiratory infections?")
if st.button("Search") and q.strip():
    try:
        t0 = time.time()
        qvec = svc["embedder"].encode([q])[0]  # list[float]
        pre = svc["retrieve_dense"](svc["client"], qvec, topk=k)
        t1 = time.time()

        post = pre
        t_r = 0.0
        if reranker == "MiniCOIL":
            t2 = time.time()
            post = svc["rr_minicoil"].rerank(q, pre)
            t3 = time.time(); t_r = (t3 - t2) * 1000.0
        elif reranker == "ColBERT":
            t2 = time.time()
            post = svc["rr_colbert"].rerank(q, pre)
            t3 = time.time(); t_r = (t3 - t2) * 1000.0

        recall_ms = (t1 - t0) * 1000.0
        total_ms = recall_ms + t_r

        st.subheader("Latency")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recall (ms)", f"{recall_ms:.1f}")
        c2.metric("Rerank (ms)", f"{t_r:.1f}")
        c3.metric("Total (ms)", f"{total_ms:.1f}")

        def fmt(rows):
            out = []
            for i, r in enumerate(rows[:show_n], 1):
                txt = r.get("text", "")
                snip = txt if show_text else (txt[:200] + "…")*(len(txt) > 200)
                out.append(f"{i:2d}. ({r['score']:.3f}) — {snip}")
            return "\n".join(out) if out else "(no results)"

        col1, col2 = st.columns(2)
        col1.markdown("#### Before (dense recall)")
        col1.code(fmt(pre), language="text")

        title = "After (reranked: **None**)" if reranker == "None" else f"After (reranked: **{reranker}**)"
        col2.markdown(f"#### {title}")
        col2.code(fmt(post), language="text")

        with st.expander("Show IDs"):
            st.write("Before:", [r["id"] for r in pre[:show_n]])
            st.write("After:", [r["id"] for r in post[:show_n]])

    except Exception as e:
        st.error("Query failed:")
        st.exception(e)
else:
    st.info("Enter a query and click **Search**.")
