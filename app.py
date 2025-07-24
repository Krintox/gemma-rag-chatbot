import streamlit as st
from rag_pipeline import build_index, ask

st.title("💬 Personal Dev Assistant (Offline RAG)")

if st.button("🔄 Build Index from Notes"):
    build_index()
    st.success("Index built from your personal notes!")

query = st.text_input("Ask a question based on your notes:")
if query:
    response = ask(query)
    st.write(response.response)
