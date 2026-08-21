"""Streamlit UI.

Run with: streamlit run app.py
"""
import tempfile
import os

import streamlit as st

from src.pipeline import ingest, answer
from src.embed_store import VectorStore
from src.retrieve import retrieve

st.set_page_config(page_title="Document QA", layout="wide")
st.title("Multimodal document QA")
st.caption("Upload a PDF with tables and figures, then ask questions grounded in them.")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Ingest a document")
    uploaded = st.file_uploader("PDF", type=["pdf"])
    if uploaded and st.button("Index this document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        with st.spinner("Extracting, captioning tables/figures, embedding..."):
            ingest(tmp_path, store=st.session_state.store)
        os.unlink(tmp_path)
        st.success(f"Indexed. {st.session_state.store.count()} chunks in the index.")

    st.metric("Chunks indexed", st.session_state.store.count())

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for img in msg.get("images", []):
            st.image(img, width=320)

question = st.chat_input("Ask a question about the indexed document(s)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            items = retrieve(st.session_state.store, question)
            reply = answer(question, store=st.session_state.store)
        st.markdown(reply)
        images = [it.image_path for it in items if it.type in ("table", "figure") and it.image_path]
        for img in images:
            if os.path.exists(img):
                st.image(img, width=320, caption="Retrieved source")
    st.session_state.messages.append({"role": "assistant", "content": reply, "images": images})
