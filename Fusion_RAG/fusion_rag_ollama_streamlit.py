# -*- coding: utf-8 -*-
# Fusion RAG with Ollama and Streamlit

import os
import numpy as np
from rank_bm25 import BM25Okapi
import fitz
import requests
import re
import json
import time
import streamlit as st
import tempfile
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def embeddings_create(self, model, input_text):
        """Create embeddings using Ollama"""
        if isinstance(input_text, str):
            input_text = [input_text]
        
        embeddings = []
        for text in input_text:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text}
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    st.error(f"Failed to create embedding: {response.text}")
                    return None
            except Exception as e:
                st.error(f"Error creating embedding: {str(e)}")
                return None
        
        return embeddings
    
    def chat_completion_create(self, model, messages, temperature=0):
        """Create chat completion using Ollama"""
        # Convert messages to Ollama format
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        prompt += "Assistant: "
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                st.error(f"Failed to generate response: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

# Initialize Ollama client
client = OllamaClient(OLLAMA_BASE_URL)

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        
        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "start_char": i,
                    "end_char": i + len(chunk)
                }
            }
            chunks.append(chunk_data)
    
    return chunks

def clean_text(text):
    """Clean text by removing extra whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')
    text = ' '.join(text.split())
    return text

def create_embeddings(texts, model="nomic-embed-text"):
    """Create embeddings for the given texts using Ollama."""
    input_texts = texts if isinstance(texts, list) else [texts]
    
    # Process in smaller batches to avoid overwhelming Ollama
    batch_size = 10
    all_embeddings = []
    
    progress_bar = st.progress(0)
    total_batches = (len(input_texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        
        batch_embeddings = client.embeddings_create(model, batch)
        if not batch_embeddings:
            return None
        
        all_embeddings.extend(batch_embeddings)
        
        # Update progress
        progress = (i // batch_size + 1) / total_batches
        progress_bar.progress(progress)
    
    progress_bar.empty()
    
    if isinstance(texts, str):
        return all_embeddings[0]
    
    return all_embeddings

class SimpleVectorStore:
    """A simple vector store implementation using NumPy."""
    
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        """Add an item to the vector store."""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def add_items(self, items, embeddings):
        """Add multiple items to the vector store."""
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata={**item.get("metadata", {}), "index": i}
            )
    
    def similarity_search_with_scores(self, query_embedding, k=5):
        """Find the most similar items to a query embedding with similarity scores."""
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        
        return results
    
    def get_all_documents(self):
        """Get all documents in the store."""
        return [{"text": text, "metadata": meta} for text, meta in zip(self.texts, self.metadata)]

def create_bm25_index(chunks):
    """Create a BM25 index from the given chunks."""
    texts = [chunk["text"] for chunk in chunks]
    tokenized_docs = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

def bm25_search(bm25, chunks, query, k=5):
    """Search the BM25 index with a query."""
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    
    results = []
    for i, score in enumerate(scores):
        metadata = chunks[i].get("metadata", {}).copy()
        metadata["index"] = i
        
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,
            "bm25_score": float(score)
        })
    
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results[:k]

def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """Perform fusion retrieval combining vector-based and BM25 search."""
    epsilon = 1e-8
    
    # Get vector search results
    query_embedding = create_embeddings(query)
    if not query_embedding:
        return []
    
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))
    
    # Create dictionaries to map document index to score
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    
    # Ensure all documents have scores for both methods
    all_docs = vector_store.get_all_documents()
    combined_results = []
    
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)
        bm25_score = bm25_scores_dict.get(i, 0.0)
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    
    # Extract scores as arrays
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    
    # Normalize scores
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    
    # Compute combined scores
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores
    
    # Add combined scores to results
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    
    # Sort by combined score (descending)
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return combined_results[:k]

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Process a document for fusion retrieval."""
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return None, None, None
    
    with st.spinner("Cleaning and chunking text..."):
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
    
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    with st.spinner(f"Creating embeddings for {len(chunks)} chunks..."):
        embeddings = create_embeddings(chunk_texts)
        if not embeddings:
            return None, None, None
    
    with st.spinner("Building vector store..."):
        vector_store = SimpleVectorStore()
        vector_store.add_items(chunks, embeddings)
    
    with st.spinner("Creating BM25 index..."):
        bm25_index = create_bm25_index(chunks)
    
    return chunks, vector_store, bm25_index

def generate_response(query, context, model="llama3"):
    """Generate a response based on the query and context."""
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
    If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""

    user_prompt = f"""Context:
    {context}

    Question: {query}

    Please answer the question based on the provided context."""

    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        return response if response else "Sorry, I couldn't generate a response."
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, there was an error generating the response."

def answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, model="llama3"):
    """Answer a query using fusion RAG."""
    retrieved_docs = fusion_retrieval(query, chunks, vector_store, bm25_index, k=k, alpha=alpha)
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    response = generate_response(query, context, model)
    
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

def vector_only_rag(query, vector_store, k=5, model="llama3"):
    """Answer a query using only vector-based RAG."""
    query_embedding = create_embeddings(query)
    if not query_embedding:
        return {"query": query, "retrieved_documents": [], "response": "Error creating query embedding"}
    
    retrieved_docs = vector_store.similarity_search_with_scores(query_embedding, k=k)
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    response = generate_response(query, context, model)
    
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

def bm25_only_rag(query, chunks, bm25_index, k=5, model="llama3"):
    """Answer a query using only BM25-based RAG."""
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    response = generate_response(query, context, model)
    
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

def compare_retrieval_methods(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, model="llama3"):
    """Compare different retrieval methods for a query."""
    
    with st.spinner("Running vector-only RAG..."):
        vector_result = vector_only_rag(query, vector_store, k, model)
    
    with st.spinner("Running BM25-only RAG..."):
        bm25_result = bm25_only_rag(query, chunks, bm25_index, k, model)
    
    with st.spinner("Running fusion RAG..."):
        fusion_result = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k, alpha, model)
    
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result
    }

def create_score_visualization(fusion_docs):
    """Create a visualization of fusion scores."""
    if not fusion_docs:
        return None
    
    # Extract scores for visualization
    doc_indices = [f"Doc {i+1}" for i in range(len(fusion_docs))]
    vector_scores = [doc.get("vector_score", 0) for doc in fusion_docs]
    bm25_scores = [doc.get("bm25_score", 0) for doc in fusion_docs]
    combined_scores = [doc.get("combined_score", 0) for doc in fusion_docs]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "Document": doc_indices,
        "Vector Score": vector_scores,
        "BM25 Score": bm25_scores,
        "Combined Score": combined_scores
    })
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Vector Score',
        x=df['Document'],
        y=df['Vector Score'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='BM25 Score',
        x=df['Document'],
        y=df['BM25 Score'],
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Bar(
        name='Combined Score',
        x=df['Document'],
        y=df['Combined Score'],
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Retrieval Scores Comparison',
        xaxis_title='Documents',
        yaxis_title='Scores',
        barmode='group',
        height=400
    )
    
    return fig

def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except:
        return False

def get_available_models():
    """Get list of available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

def main():
    st.set_page_config(
        page_title="Fusion RAG with Ollama",
        page_icon="🔀",
        layout="wide"
    )
    
    st.title("🔀 Fusion RAG with Ollama")
    st.markdown("Combines vector-based and keyword-based retrieval for better results using locally-hosted language models.")
    
    # Check Ollama connection
    if not check_ollama_connection():
        st.error("❌ Cannot connect to Ollama. Please make sure Ollama is running on http://localhost:11434")
        st.info("To start Ollama: `ollama serve`")
        return
    
    st.success("✅ Connected to Ollama")
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        st.error("No models found in Ollama. Please pull some models first.")
        st.info("Example: `ollama pull llama3` and `ollama pull nomic-embed-text`")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        chat_models = [m for m in available_models if not any(embed in m.lower() for embed in ['embed', 'embedding'])]
        embed_models = [m for m in available_models if any(embed in m.lower() for embed in ['embed', 'embedding'])]
        
        if not embed_models:
            st.error("No embedding models found. Please install an embedding model like 'nomic-embed-text'")
            return
        
        selected_chat_model = st.selectbox("💬 Chat Model", chat_models, index=0 if chat_models else 0)
        selected_embed_model = st.selectbox("🔍 Embedding Model", embed_models, index=0 if embed_models else 0)
        
        # Document processing parameters
        st.subheader("📄 Document Processing")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        
        # Retrieval parameters
        st.subheader("🎯 Retrieval Settings")
        k_docs = st.slider("Documents to Retrieve", 2, 10, 5)
        alpha = st.slider("Fusion Weight (α)", 0.0, 1.0, 0.5, 0.1, 
                         help="0 = BM25 only, 1 = Vector only, 0.5 = Equal weight")
        
        # Analysis mode
        st.subheader("📊 Analysis Mode")
        comparison_mode = st.checkbox("Compare Retrieval Methods", 
                                    help="Compare Vector, BM25, and Fusion approaches")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to create a knowledge base"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"📁 {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process document
            if st.button("🔄 Process Document", type="primary"):
                chunks, vector_store, bm25_index = process_document(
                    tmp_file_path, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                
                if chunks and vector_store and bm25_index:
                    st.session_state.chunks = chunks
                    st.session_state.vector_store = vector_store
                    st.session_state.bm25_index = bm25_index
                    st.session_state.doc_processed = True
                    st.success(f"✅ Document processed! Created {len(chunks)} chunks with vector and BM25 indices.")
                else:
                    st.error("❌ Failed to process document.")
            
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    with col2:
        st.header("💬 Query Interface")
        
        if hasattr(st.session_state, 'doc_processed') and st.session_state.doc_processed:
            query = st.text_input(
                "🔍 Enter your question:",
                placeholder="What would you like to know about the document?"
            )
            
            if st.button("🚀 Get Answer", type="primary") and query:
                if comparison_mode:
                    # Compare different retrieval methods
                    st.subheader("📊 Comparison Results")
                    
                    results = compare_retrieval_methods(
                        query, 
                        st.session_state.chunks,
                        st.session_state.vector_store, 
                        st.session_state.bm25_index,
                        k=k_docs,
                        alpha=alpha,
                        model=selected_chat_model
                    )
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["🔀 Fusion", "🎯 Vector", "🔤 BM25", "📈 Analysis"])
                    
                    with tab1:
                        st.markdown("**🔀 Fusion RAG Response:**")
                        st.write(results["fusion_result"]["response"])
                        
                        if results["fusion_result"]["retrieved_documents"]:
                            with st.expander("📚 Retrieved Documents (Fusion)"):
                                for i, doc in enumerate(results["fusion_result"]["retrieved_documents"]):
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Vector", f"{doc.get('vector_score', 0):.3f}")
                                    with col_b:
                                        st.metric("BM25", f"{doc.get('bm25_score', 0):.3f}")
                                    with col_c:
                                        st.metric("Combined", f"{doc.get('combined_score', 0):.3f}")
                                    
                                    st.write(f"**Document {i+1}:**")
                                    preview = doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"]
                                    st.write(preview)
                                    st.divider()
                    
                    with tab2:
                        st.markdown("**🎯 Vector-Only RAG Response:**")
                        st.write(results["vector_result"]["response"])
                        
                        if results["vector_result"]["retrieved_documents"]:
                            with st.expander("📚 Retrieved Documents (Vector)"):
                                for i, doc in enumerate(results["vector_result"]["retrieved_documents"]):
                                    st.write(f"**Document {i+1}** (Similarity: {doc['similarity']:.3f})")
                                    preview = doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"]
                                    st.write(preview)
                                    st.divider()
                    
                    with tab3:
                        st.markdown("**🔤 BM25-Only RAG Response:**")
                        st.write(results["bm25_result"]["response"])
                        
                        if results["bm25_result"]["retrieved_documents"]:
                            with st.expander("📚 Retrieved Documents (BM25)"):
                                for i, doc in enumerate(results["bm25_result"]["retrieved_documents"]):
                                    st.write(f"**Document {i+1}** (BM25 Score: {doc['bm25_score']:.3f})")
                                    preview = doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"]
                                    st.write(preview)
                                    st.divider()
                    
                    with tab4:
                        st.markdown("**📈 Score Analysis**")
                        
                        # Create visualization
                        if results["fusion_result"]["retrieved_documents"]:
                            fig = create_score_visualization(results["fusion_result"]["retrieved_documents"])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Display fusion weights
                        st.info(f"**Fusion Configuration:** α = {alpha} (Vector weight: {alpha:.1f}, BM25 weight: {1-alpha:.1f})")
                
                else:
                    # Single fusion RAG response
                    with st.spinner("Generating fusion RAG response..."):
                        result = answer_with_fusion_rag(
                            query, 
                            st.session_state.chunks,
                            st.session_state.vector_store, 
                            st.session_state.bm25_index,
                            k=k_docs,
                            alpha=alpha,
                            model=selected_chat_model
                        )
                    
                    st.subheader(f"🔀 Fusion RAG Response")
                    st.write(result["response"])
                    
                    # Show retrieved documents with fusion scores
                    if result["retrieved_documents"]:
                        with st.expander("📚 Retrieved Documents with Fusion Scores"):
                            for i, doc in enumerate(result["retrieved_documents"]):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Vector Score", f"{doc.get('vector_score', 0):.3f}")
                                with col_b:
                                    st.metric("BM25 Score", f"{doc.get('bm25_score', 0):.3f}")
                                with col_c:
                                    st.metric("Combined Score", f"{doc.get('combined_score', 0):.3f}")
                                
                                st.write(f"**Document {i+1}:**")
                                preview = doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"]
                                st.write(preview)
                                st.divider()
                        
                        # Show score visualization
                        fig = create_score_visualization(result["retrieved_documents"])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Please upload and process a PDF document first.")
    
    # Information and examples
    if hasattr(st.session_state, 'doc_processed') and st.session_state.doc_processed:
        with st.expander("ℹ️ About Fusion RAG"):
            st.markdown("""
            **Fusion RAG** combines two complementary retrieval approaches:
            
            - **🎯 Vector-based Retrieval**: Uses semantic similarity via embeddings
            - **🔤 BM25 Retrieval**: Uses keyword matching and term frequency
            
            **Benefits:**
            - Better handling of both semantic and exact keyword matches
            - Improved recall and precision
            - Balances strengths of both approaches
            
            **Fusion Weight (α):**
            - `α = 0`: Pure BM25 (keyword-based)
            - `α = 0.5`: Equal weight (recommended)
            - `α = 1`: Pure vector-based (semantic)
            """)
        
        with st.expander("💡 Query Examples"):
            st.markdown("""
            **Try these example queries:**
            
            **Factual Questions:**
            - What is machine learning?
            - When was the concept of AI introduced?
            - Who developed the transformer architecture?
            
            **Technical Questions:**
            - How do neural networks work?
            - What are the types of machine learning algorithms?
            - Explain the attention mechanism
            
            **Comparative Questions:**
            - What's the difference between supervised and unsupervised learning?
            - Compare CNNs and RNNs
            - Vector search vs keyword search advantages
            """)

if __name__ == "__main__":
    main()


