"""
Phase 5.2: Query dashboard. Ask questions, see the answer with
citations, retrieved chunks ranked by relevance, confidence breakdown,
and a toggle to compare hybrid vs. dense-only retrieval side by side.

Run with: streamlit run frontend/dashboard.py
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vectorstore import VectorStore
from app.bm25_index import BM25Index
from app.pipeline import ask

st.set_page_config(page_title="RAG Pipeline Dashboard", layout="wide")
st.title("RAG Pipeline — Hybrid Search Dashboard")


@st.cache_resource
def get_store_and_index():
    store = VectorStore()
    bm25 = BM25Index()
    try:
        bm25.load()
    except FileNotFoundError:
        bm25.build(store.all_chunks())
    return store, bm25


store, bm25 = get_store_and_index()

docs = store.list_documents()
st.sidebar.header("Indexed Documents")
if docs:
    for d in docs:
        st.sidebar.write(f"📄 {os.path.basename(d['source_file'])} — {d['chunk_count']} chunks")
else:
    st.sidebar.warning("No documents indexed yet. Run scripts/ingest_sample_docs.py first.")

compare_mode = st.sidebar.checkbox("Compare hybrid vs. dense-only", value=False)
verify = st.sidebar.checkbox("Verify citations (LLM-as-judge)", value=True)

question = st.text_input("Ask a question about the indexed docs:")

# FIX: previously `result` only lived inside the `if st.button(...)` block,
# so it vanished the moment Streamlit reran the script for ANY other
# widget interaction (e.g. opening a chunk expander, toggling compare
# mode) — the button is no longer "just clicked" on that rerun, so the
# whole answer section disappeared. Storing it in session_state makes it
# survive reruns until a new question is actually asked.
if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating..."):
        st.session_state["result"] = ask(question, store, bm25, use_hybrid=True, verify=verify)
        st.session_state["question"] = question

result = st.session_state.get("result")

if result:
    st.subheader("Answer")
    st.write(result["answer"])

    conf = result["confidence"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval confidence", f"{conf['retrieval_confidence']*100:.0f}%")
    c2.metric("Citation accuracy", f"{conf['citation_accuracy']*100:.0f}%" if conf['citation_accuracy'] is not None else "N/A")
    c3.metric("Completeness", f"{conf['answer_completeness']*100:.0f}%")
    c4.metric("Composite", f"{conf['composite_confidence']*100:.0f}%")

    if conf["is_low_confidence"]:
        st.warning("⚠️ Low confidence — treat this answer with caution.")

    st.subheader("Retrieved Chunks (Hybrid, Reranked)")
    for i, c in enumerate(result.get("retrieved_chunks", []), start=1):
        meta = c.get("metadata", {})
        with st.expander(f"[{i}] {meta.get('source_file', 'unknown')} — {meta.get('section_heading', 'N/A')} (rerank score: {c.get('rerank_score', 0):.3f})"):
            st.write(c["text"])

    if result.get("citation_report"):
        st.subheader("Citation Verification")
        for check in result["citation_report"]["citation_checks"]:
            icon = "✅" if check["supported"] else "❌"
            st.write(f"{icon} [{check['citation_index']}] \"{check['claim']}\" — {check['reason']}")

    if compare_mode:
        st.subheader("Hybrid vs. Dense-only Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Hybrid (dense + sparse + RRF)**")
            for c in result.get("retrieved_chunks", []):
                st.caption(f"{c.get('metadata', {}).get('section_heading', 'N/A')} — score {c.get('rerank_score', 0):.3f}")
        with col2:
            st.markdown("**Dense-only**")
            for c in result.get("dense_only_chunks", []):
                st.caption(f"{c.get('metadata', {}).get('section_heading', 'N/A')} — score {c.get('rerank_score', 0):.3f}")