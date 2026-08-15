"""Streamlit web interface for Souprise.

Run via `souprise gui` or directly:
    streamlit run souprise/gui/app.py

Everything happens locally: index loading, search, and generation.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import os
import time
from pathlib import Path

import streamlit as st

from souprise import RAGConfig, SimpleHDCRetriever, SoupriseRAG
from souprise.core.pipeline import DEFAULT_MODELS, resolve_backend

_LOGO = str(Path(__file__).parent / "assets" / "logo.svg")

st.set_page_config(page_title="Souprise", page_icon=None, layout="wide")
st.logo(_LOGO, size="large")

st.image(_LOGO, width=300)
st.caption("Ask your business records. Everything runs on this machine.")

with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Index source",
        ["Synthetic demo data", "Index file"],
        help="Load a persistent index built with 'souprise index build', "
             "or generate demo data to explore.",
    )
    if source == "Index file":
        index_path = st.text_input("Index file", value="souprise_index.db")
        data_size = None
    else:
        index_path = None
        data_size = st.slider("Demo records", 1_000, 50_000, 10_000, step=1_000)

    st.header("Answering")
    search_only = st.checkbox(
        "Search only (no language model)",
        value=True,
        help="Show the matching records without loading an LLM. "
             "Uncheck to generate written answers with a local model.",
    )
    model_path = st.text_input(
        "Model (local path or Hugging Face ID)",
        value=DEFAULT_MODELS[resolve_backend("auto")],
        disabled=search_only,
    )
    k = st.slider("Records per answer", 1, 10, 5)

    if st.button("Load", type="primary"):
        with st.spinner("Loading index..."):
            rag = SoupriseRAG(RAGConfig(retriever="simple",
                                        model_path=model_path, retrieval_k=k))
            if index_path:
                if not os.path.exists(index_path):
                    st.error(f"No index at {index_path}. Build one with "
                             "'souprise index build'.")
                    st.stop()
                rag.retriever = SimpleHDCRetriever.load(index_path)
            else:
                rag.index_from_business_data(n=data_size, seed=42)
        if not search_only:
            with st.spinner("Loading model (first run downloads it once)..."):
                rag.load_model()
        st.session_state["rag"] = rag
        st.session_state["search_only"] = search_only
        st.success(f"Ready: {rag._get_retriever().size:,} records indexed.")

if "rag" not in st.session_state:
    st.info("Choose a data source on the left and press Load.")
    st.stop()

rag = st.session_state["rag"]
question = st.text_input(
    "Your question",
    placeholder="Which invoices are overdue?",
)

if question:
    if st.session_state["search_only"]:
        start = time.perf_counter()
        results = rag._get_retriever().search(question, k=k)
        latency = (time.perf_counter() - start) * 1000
        st.metric("Retrieval", f"{latency:.1f} ms")
    else:
        result = rag.query(question, k=k)
        st.subheader("Answer")
        st.write(result.answer)
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval", f"{result.retrieval_latency * 1000:.1f} ms")
        col2.metric("Generation", f"{result.generation_latency * 1000:.1f} ms")
        col3.metric("Total", f"{result.total_latency * 1000:.1f} ms")
        results = result.retrieval_results

    st.subheader("Source records")
    for r in results:
        with st.expander(f"{r.score:.3f} · {r.title}"):
            st.text(r.content)
            if r.metadata.get("tags"):
                st.caption("Tags: " + ", ".join(r.metadata["tags"]))
