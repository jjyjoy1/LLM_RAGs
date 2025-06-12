import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from lightrag import LightRAG, QueryParam

st.title("LightRAG with Local LLMs")

# Initialize LightRAG (same as above)
@st.cache_resource
def init_lightrag():
    return LightRAG(
        working_dir="./lightrag_workdir",
        llm_model_func=ollama_model_complete,
        llm_model_name='llama3.1:8b',
        llm_model_kwargs={"options": {"num_ctx": 32768}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text")
        ),
    )

rag = init_lightrag()

# File upload
uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
if uploaded_file:
    content = str(uploaded_file.read(), "utf-8")
    with st.spinner("Processing document..."):
        rag.insert(content)
    st.success("Document processed!")

# Query interface
query = st.text_input("Ask a question about your documents:")
search_mode = st.selectbox("Search Mode", ["global", "local", "hybrid", "naive"])

if query:
    with st.spinner("Searching..."):
        result = rag.query(query, param=QueryParam(mode=search_mode))
        st.write("**Answer:**")
        st.write(result)
