# -*- coding: utf-8 -*-
# Adaptive RAG with Ollama and Streamlit

import os
import numpy as np
import json
import fitz
import requests
import re
import streamlit as st
import tempfile
from pathlib import Path
import time

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
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text}
            )
            if response.status_code == 200:
                embeddings.append(response.json()["embedding"])
            else:
                st.error(f"Failed to create embedding: {response.text}")
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

# Initialize Ollama client
client = OllamaClient(OLLAMA_BASE_URL)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        mypdf = fitz.open(pdf_path)
        all_text = ""
        
        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            text = page.get_text("text")
            all_text += text
        
        mypdf.close()
        return all_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text, n, overlap):
    """Chunk text into segments with overlap."""
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks

class SimpleVectorStore:
    """Simple vector store implementation using NumPy."""
    
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        """Add an item to the vector store."""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """Find the most similar items to a query embedding."""
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            if filter_func and not filter_func(self.metadata[i]):
                continue
            
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        
        return results

def create_embeddings(text, model="nomic-embed-text"):
    """Create embeddings using Ollama."""
    try:
        if isinstance(text, list):
            embeddings = client.embeddings_create(model, text)
            return embeddings
        else:
            embeddings = client.embeddings_create(model, [text])
            return embeddings[0] if embeddings else None
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Process a document for use with adaptive retrieval."""
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            return None, None
    
    with st.spinner("Chunking text..."):
        chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    
    with st.spinner(f"Creating embeddings for {len(chunks)} chunks..."):
        chunk_embeddings = create_embeddings(chunks)
        if not chunk_embeddings:
            return None, None
    
    store = SimpleVectorStore()
    
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    
    return chunks, store

def classify_query(query, model="llama3"):
    """Classify a query into one of four categories."""
    system_prompt = """You are an expert at classifying questions. 
        Classify the given query into exactly one of these categories:
        - Factual: Queries seeking specific, verifiable information.
        - Analytical: Queries requiring comprehensive analysis or explanation.
        - Opinion: Queries about subjective matters or seeking diverse viewpoints.
        - Contextual: Queries that depend on user-specific context.

        Return ONLY the category name, without any explanation or additional text.
    """
    
    user_prompt = f"Classify this query: {query}"
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        if response:
            category = response.strip()
            valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
            
            for valid in valid_categories:
                if valid in category:
                    return valid
        
        return "Factual"  # Default fallback
    except Exception as e:
        st.error(f"Error classifying query: {str(e)}")
        return "Factual"

def factual_retrieval_strategy(query, vector_store, k=4, model="llama3"):
    """Retrieval strategy for factual queries focusing on precision."""
    system_prompt = """You are an expert at enhancing search queries.
        Your task is to reformulate the given factual query to make it more precise and 
        specific for information retrieval. Focus on key entities and their relationships.

        Provide ONLY the enhanced query without any explanation.
    """
    
    user_prompt = f"Enhance this factual query: {query}"
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        enhanced_query = response.strip() if response else query
        
        query_embedding = create_embeddings(enhanced_query)
        if not query_embedding:
            return []
        
        initial_results = vector_store.similarity_search(query_embedding, k=k*2)
        
        ranked_results = []
        for doc in initial_results:
            relevance_score = score_document_relevance(enhanced_query, doc["text"], model)
            ranked_results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity": doc["similarity"],
                "relevance_score": relevance_score
            })
        
        ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return ranked_results[:k]
    
    except Exception as e:
        st.error(f"Error in factual retrieval: {str(e)}")
        return []

def analytical_retrieval_strategy(query, vector_store, k=4, model="llama3"):
    """Retrieval strategy for analytical queries focusing on comprehensive coverage."""
    system_prompt = """You are an expert at breaking down complex questions.
    Generate sub-questions that explore different aspects of the main analytical query.
    These sub-questions should cover the breadth of the topic and help retrieve 
    comprehensive information.

    Return a list of exactly 3 sub-questions, one per line.
    """
    
    user_prompt = f"Generate sub-questions for this analytical query: {query}"
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        if response:
            sub_queries = response.strip().split('\n')
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
        else:
            sub_queries = [query]
        
        all_results = []
        for sub_query in sub_queries:
            sub_query_embedding = create_embeddings(sub_query)
            if sub_query_embedding:
                results = vector_store.similarity_search(sub_query_embedding, k=2)
                all_results.extend(results)
        
        # Remove duplicates and ensure diversity
        unique_texts = set()
        diverse_results = []
        
        for result in all_results:
            if result["text"] not in unique_texts:
                unique_texts.add(result["text"])
                diverse_results.append(result)
        
        return diverse_results[:k]
    
    except Exception as e:
        st.error(f"Error in analytical retrieval: {str(e)}")
        return []

def opinion_retrieval_strategy(query, vector_store, k=4, model="llama3"):
    """Retrieval strategy for opinion queries focusing on diverse perspectives."""
    system_prompt = """You are an expert at identifying different perspectives on a topic.
        For the given query about opinions or viewpoints, identify different perspectives 
        that people might have on this topic.

        Return a list of exactly 3 different viewpoint angles, one per line.
    """
    
    user_prompt = f"Identify different perspectives on: {query}"
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        if response:
            viewpoints = response.strip().split('\n')
            viewpoints = [v.strip() for v in viewpoints if v.strip()]
        else:
            viewpoints = [query]
        
        all_results = []
        for viewpoint in viewpoints:
            combined_query = f"{query} {viewpoint}"
            viewpoint_embedding = create_embeddings(combined_query)
            if viewpoint_embedding:
                results = vector_store.similarity_search(viewpoint_embedding, k=2)
                for result in results:
                    result["viewpoint"] = viewpoint
                all_results.extend(results)
        
        # Select diverse range of opinions
        selected_results = []
        for viewpoint in viewpoints:
            viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
            if viewpoint_docs:
                selected_results.append(viewpoint_docs[0])
        
        # Fill remaining slots
        remaining_slots = k - len(selected_results)
        if remaining_slots > 0:
            remaining_docs = [r for r in all_results if r not in selected_results]
            remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
            selected_results.extend(remaining_docs[:remaining_slots])
        
        return selected_results[:k]
    
    except Exception as e:
        st.error(f"Error in opinion retrieval: {str(e)}")
        return []

def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None, model="llama3"):
    """Retrieval strategy for contextual queries integrating user context."""
    if not user_context:
        system_prompt = """You are an expert at understanding implied context in questions.
For the given query, infer what contextual information might be relevant or implied 
but not explicitly stated. Focus on what background would help answering this query.

Return a brief description of the implied context."""
        
        user_prompt = f"Infer the implied context in this query: {query}"
        
        try:
            response = client.chat_completion_create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            user_context = response.strip() if response else ""
        except:
            user_context = ""
    
    # Reformulate query with context
    system_prompt = """You are an expert at reformulating questions with context.
    Given a query and some contextual information, create a more specific query that 
    incorporates the context to get more relevant information.

    Return ONLY the reformulated query without explanation."""
    
    user_prompt = f"""
    Query: {query}
    Context: {user_context}

    Reformulate the query to incorporate this context:"""
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        contextualized_query = response.strip() if response else query
        
        query_embedding = create_embeddings(contextualized_query)
        if not query_embedding:
            return []
        
        initial_results = vector_store.similarity_search(query_embedding, k=k*2)
        
        ranked_results = []
        for doc in initial_results:
            context_relevance = score_document_context_relevance(query, user_context, doc["text"], model)
            ranked_results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity": doc["similarity"],
                "context_relevance": context_relevance
            })
        
        ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
        return ranked_results[:k]
    
    except Exception as e:
        st.error(f"Error in contextual retrieval: {str(e)}")
        return []

def score_document_relevance(query, document, model="llama3"):
    """Score document relevance to a query using LLM."""
    system_prompt = """You are an expert at evaluating document relevance.
        Rate the relevance of a document to a query on a scale from 0 to 10, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query

        Return ONLY a numerical score between 0 and 10, nothing else.
    """
    
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    user_prompt = f"""
        Query: {query}

        Document: {doc_preview}

        Relevance score (0-10):
    """
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        if response:
            score_text = response.strip()
            match = re.search(r'(\d+(\.\d+)?)', score_text)
            if match:
                score = float(match.group(1))
                return min(10, max(0, score))
        
        return 5.0
    except:
        return 5.0

def score_document_context_relevance(query, context, document, model="llama3"):
    """Score document relevance considering both query and context."""
    system_prompt = """You are an expert at evaluating document relevance considering context.
        Rate the document on a scale from 0 to 10 based on how well it addresses the query
        when considering the provided context, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query in the given context

        Return ONLY a numerical score between 0 and 10, nothing else.
    """
    
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    user_prompt = f"""
    Query: {query}
    Context: {context}

    Document: {doc_preview}

    Relevance score considering context (0-10):
    """
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        if response:
            score_text = response.strip()
            match = re.search(r'(\d+(\.\d+)?)', score_text)
            if match:
                score = float(match.group(1))
                return min(10, max(0, score))
        
        return 5.0
    except:
        return 5.0

def adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """Perform adaptive retrieval by selecting and executing the appropriate strategy."""
    query_type = classify_query(query)
    
    if query_type == "Factual":
        results = factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "Analytical":
        results = analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "Opinion":
        results = opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "Contextual":
        results = contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        results = factual_retrieval_strategy(query, vector_store, k)
    
    return results, query_type

def generate_response(query, results, query_type, model="llama3"):
    """Generate a response based on query, retrieved documents, and query type."""
    context = "\n\n---\n\n".join([r["text"] for r in results])
    
    if query_type == "Factual":
        system_prompt = """You are a helpful assistant providing factual information.
    Answer the question based on the provided context. Focus on accuracy and precision.
    If the context doesn't contain the information needed, acknowledge the limitations."""
        
    elif query_type == "Analytical":
        system_prompt = """You are a helpful assistant providing analytical insights.
    Based on the provided context, offer a comprehensive analysis of the topic.
    Cover different aspects and perspectives in your explanation.
    If the context has gaps, acknowledge them while providing the best analysis possible."""
        
    elif query_type == "Opinion":
        system_prompt = """You are a helpful assistant discussing topics with multiple viewpoints.
    Based on the provided context, present different perspectives on the topic.
    Ensure fair representation of diverse opinions without showing bias.
    Acknowledge where the context presents limited viewpoints."""
        
    elif query_type == "Contextual":
        system_prompt = """You are a helpful assistant providing contextually relevant information.
    Answer the question considering both the query and its context.
    Make connections between the query context and the information in the provided documents.
    If the context doesn't fully address the specific situation, acknowledge the limitations."""
        
    else:
        system_prompt = """You are a helpful assistant. Answer the question based on the provided context. If you cannot answer from the context, acknowledge the limitations."""
    
    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    Please provide a helpful response based on the context.
    """
    
    try:
        response = client.chat_completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        return response if response else "Sorry, I couldn't generate a response."
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, there was an error generating the response."

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

# Streamlit App
def main():
    st.set_page_config(
        page_title="Adaptive RAG with Ollama",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Adaptive RAG with Ollama")
    st.markdown("Upload a PDF document and ask questions using locally-hosted language models via Ollama.")
    
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
        st.header("Configuration")
        
        # Model selection
        chat_models = [m for m in available_models if not any(embed in m.lower() for embed in ['embed', 'embedding'])]
        embed_models = [m for m in available_models if any(embed in m.lower() for embed in ['embed', 'embedding'])]
        
        if not embed_models:
            st.error("No embedding models found. Please install an embedding model like 'nomic-embed-text'")
            return
        
        selected_chat_model = st.selectbox("Chat Model", chat_models, index=0 if chat_models else 0)
        selected_embed_model = st.selectbox("Embedding Model", embed_models, index=0 if embed_models else 0)
        
        # Chunking parameters
        st.subheader("Document Processing")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        k_docs = st.slider("Documents to Retrieve", 2, 10, 4)
        
        # User context
        st.subheader("Optional Context")
        user_context = st.text_area("Provide additional context for your queries (optional)")
    
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
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process document
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    chunks, vector_store = process_document(
                        tmp_file_path, 
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    
                    if chunks and vector_store:
                        st.session_state.chunks = chunks
                        st.session_state.vector_store = vector_store
                        st.session_state.doc_processed = True
                        st.success(f"✅ Document processed! Created {len(chunks)} chunks.")
                    else:
                        st.error("❌ Failed to process document.")
            
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    with col2:
        st.header("💬 Ask Questions")
        
        if hasattr(st.session_state, 'doc_processed') and st.session_state.doc_processed:
            query = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about the document?"
            )
            
            if st.button("Get Answer", type="primary") and query:
                with st.spinner("Generating answer..."):
                    # Update global client with selected embedding model
                    results, query_type = adaptive_retrieval(
                        query, 
                        st.session_state.vector_store, 
                        k=k_docs,
                        user_context=user_context if user_context else None
                    )
                    
                    if results:
                        response = generate_response(
                            query, 
                            results, 
                            query_type, 
                            model=selected_chat_model
                        )
                        
                        # Display results
                        st.subheader(f"🎯 Answer (Query Type: {query_type})")
                        st.write(response)
                        
                        # Show retrieved documents
                        with st.expander("📚 Retrieved Documents"):
                            for i, doc in enumerate(results):
                                st.write(f"**Document {i+1}** (Similarity: {doc['similarity']:.3f})")
                                st.write(doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"])
                                st.write("---")
                    else:
                        st.error("No relevant documents found.")
        else:
            st.info("👆 Please upload and process a PDF document first.")
    
    # Query History
    if hasattr(st.session_state, 'doc_processed') and st.session_state.doc_processed:
        with st.expander("📝 Query Examples"):
            st.markdown("""
            **Factual Questions:**
            - What is [specific concept]?
            - When did [event] happen?
            - Who is [person]?
            
            **Analytical Questions:**
            - How does [system] work?
            - What are the implications of [concept]?
            - Analyze the relationship between [A] and [B]
            
            **Opinion Questions:**
            - What are different views on [topic]?
            - Is [approach] better than [alternative]?
            - What are the pros and cons of [concept]?
            
            **Contextual Questions:**
            - How might [concept] apply to [context]?
            - What would be the impact in [scenario]?
            - How does this relate to [specific situation]?
            """)

if __name__ == "__main__":
    main()

