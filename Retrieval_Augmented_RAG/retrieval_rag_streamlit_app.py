# Enhanced RAG System with Ollama and Streamlit
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import numpy as np
import requests
import json
import time
from typing import List, Dict, Tuple, Optional
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate_embeddings(self, texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
        """Generate embeddings using Ollama"""
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)
            except Exception as e:
                st.error(f"Error generating embedding: {str(e)}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)  # Default dimension
                
        return embeddings
    
    def chat_completion(self, messages: List[Dict], model: str = "llama3.1") -> str:
        """Generate chat completion using Ollama"""
        try:
            # Convert messages to Ollama format
            prompt = self._format_messages(messages)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I couldn't generate a response due to an error."
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to Ollama prompt format"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = [model["name"] for model in response.json()["models"]]
            return models
        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
            return ["llama3.1", "mistral", "nomic-embed-text"]  # Default fallback

class RobustPDFExtractor:
    """Enhanced PDF text extraction with multiple fallback methods"""
    
    def __init__(self):
        self.use_ocr = False
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from PDF with multiple methods and robust cleaning
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Method 1: Try standard text extraction
            text = self._extract_with_pymupdf(tmp_path)
            
            # If text is too short or seems like OCR is needed, try OCR
            if len(text.strip()) < 100 or self._needs_ocr(text):
                st.info("Standard extraction yielded limited text. Attempting OCR...")
                ocr_text = self._extract_with_ocr(tmp_path)
                if len(ocr_text) > len(text):
                    text = ocr_text
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Clean and preprocess the extracted text
            cleaned_text = self._clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        doc = fitz.open(pdf_path)
        all_text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Try different extraction methods
            text = page.get_text("text")
            
            # If regular extraction fails, try other methods
            if len(text.strip()) < 50:
                # Try extracting text blocks
                blocks = page.get_text("dict")
                text = self._extract_from_blocks(blocks)
            
            all_text += text + "\n"
        
        doc.close()
        return all_text
    
    def _extract_from_blocks(self, blocks: Dict) -> str:
        """Extract text from text blocks when regular extraction fails"""
        text = ""
        for block in blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text += span.get("text", "") + " "
                text += "\n"
        return text
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR as fallback"""
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            
            for page_num in range(min(doc.page_count, 10)):  # Limit OCR to first 10 pages
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # Increase resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                ocr_text = pytesseract.image_to_string(img, lang='eng')
                all_text += ocr_text + "\n"
            
            doc.close()
            return all_text
            
        except Exception as e:
            st.warning(f"OCR extraction failed: {str(e)}")
            return ""
    
    def _needs_ocr(self, text: str) -> bool:
        """Determine if OCR is needed based on text characteristics"""
        if len(text.strip()) < 100:
            return True
        
        # Check for signs that text might be garbled or incomplete
        weird_chars = sum(1 for c in text if ord(c) > 127)
        total_chars = len(text)
        
        if total_chars > 0 and (weird_chars / total_chars) > 0.1:
            return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove weird characters and fix encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words split across lines
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        
        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely page numbers
            if re.match(r'^\d+$', line):
                continue
            # Skip very short lines that might be headers/footers
            if len(line) > 10:
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)

class EnhancedVectorStore:
    """Enhanced vector store with better similarity calculations"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vectors = []
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], vectors: List[List[float]], metadata: List[Dict]):
        """Add documents to the vector store"""
        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(np.array(vec))
            self.metadata.append(meta)
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Search for most similar documents using cosine similarity"""
        if not self.vectors:
            return []
        
        query_array = np.array(query_vector)
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            # Compute cosine similarity
            dot_product = np.dot(query_array, vector)
            norm_query = np.linalg.norm(query_array)
            norm_vector = np.linalg.norm(vector)
            
            if norm_query > 0 and norm_vector > 0:
                similarity = dot_product / (norm_query * norm_vector)
            else:
                similarity = 0.0
            
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })
        
        return results

class EnhancedRAGSystem:
    """Enhanced RAG system with Ollama integration"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_client = OllamaClient(ollama_base_url)
        self.pdf_extractor = RobustPDFExtractor()
        
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 0) -> List[str]:
        """Split text into chunks with optional overlap"""
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, pdf_file, chunk_size: int = 800, embedding_model: str = "nomic-embed-text"):
        """Process PDF document and create vector store"""
        with st.spinner("Extracting text from PDF..."):
            text = self.pdf_extractor.extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            st.error("No text could be extracted from the PDF")
            return None, None, None
        
        st.success(f"Extracted {len(text)} characters from PDF")
        
        with st.spinner("Chunking text..."):
            chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=0)
        
        st.info(f"Created {len(chunks)} chunks")
        
        with st.spinner("Generating embeddings..."):
            progress_bar = st.progress(0)
            chunk_embeddings = []
            
            # Process chunks in smaller batches for better progress tracking
            batch_size = 5
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = self.ollama_client.generate_embeddings(batch, embedding_model)
                chunk_embeddings.extend(batch_embeddings)
                
                progress = min((i + batch_size) / len(chunks), 1.0)
                progress_bar.progress(progress)
            
            progress_bar.empty()
        
        # Create vector store
        vector_store = EnhancedVectorStore()
        metadata = [{"chunk_index": i, "source": "uploaded_pdf"} for i in range(len(chunks))]
        vector_store.add_documents(chunks, chunk_embeddings, metadata)
        
        doc_info = {
            "chunks": chunks,
            "total_text_length": len(text),
            "chunk_count": len(chunks)
        }
        
        return chunks, vector_store, doc_info
    
    def calculate_chunk_values(self, query: str, chunks: List[str], vector_store: EnhancedVectorStore, 
                             embedding_model: str, irrelevant_chunk_penalty: float = 0.2) -> List[float]:
        """Calculate chunk values for RSE"""
        query_embedding = self.ollama_client.generate_embeddings([query], embedding_model)[0]
        
        num_chunks = len(chunks)
        results = vector_store.search(query_embedding, top_k=num_chunks)
        
        relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}
        
        chunk_values = []
        for i in range(num_chunks):
            score = relevance_scores.get(i, 0.0)
            value = score - irrelevant_chunk_penalty
            chunk_values.append(value)
        
        return chunk_values
    
    def find_best_segments(self, chunk_values: List[float], max_segment_length: int = 20, 
                         total_max_length: int = 30, min_segment_value: float = 0.2) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Find optimal segments using enhanced algorithm"""
        best_segments = []
        segment_scores = []
        total_included_chunks = 0
        
        while total_included_chunks < total_max_length:
            best_score = min_segment_value
            best_segment = None
            
            for start in range(len(chunk_values)):
                if any(start >= s[0] and start < s[1] for s in best_segments):
                    continue
                
                for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                    end = start + length
                    
                    if any(end > s[0] and end <= s[1] for s in best_segments):
                        continue
                    
                    segment_value = sum(chunk_values[start:end])
                    
                    if segment_value > best_score:
                        best_score = segment_value
                        best_segment = (start, end)
            
            if best_segment:
                best_segments.append(best_segment)
                segment_scores.append(best_score)
                total_included_chunks += best_segment[1] - best_segment[0]
            else:
                break
        
        best_segments = sorted(best_segments, key=lambda x: x[0])
        return best_segments, segment_scores
    
    def reconstruct_segments(self, chunks: List[str], best_segments: List[Tuple[int, int]]) -> List[Dict]:
        """Reconstruct text segments"""
        reconstructed_segments = []
        
        for start, end in best_segments:
            segment_text = " ".join(chunks[start:end])
            reconstructed_segments.append({
                "text": segment_text,
                "segment_range": (start, end),
            })
        
        return reconstructed_segments
    
    def format_segments_for_context(self, segments: List[Dict]) -> str:
        """Format segments for LLM context"""
        context = []
        
        for i, segment in enumerate(segments):
            segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
            context.append(segment_header)
            context.append(segment['text'])
            context.append("-" * 80)
        
        return "\n\n".join(context)
    
    def generate_response(self, query: str, context: str, chat_model: str) -> str:
        """Generate response using Ollama"""
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
        The context consists of document segments that have been retrieved as relevant to the user's query.
        Use the information from these segments to provide a comprehensive and accurate answer.
        If the context doesn't contain relevant information to answer the question, say so clearly."""
        
        user_prompt = f"""
Context:
{context}

Question: {query}

Please provide a helpful answer based on the context provided.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.ollama_client.chat_completion(messages, chat_model)
    
    def rag_with_rse(self, chunks: List[str], vector_store: EnhancedVectorStore, query: str, 
                     embedding_model: str, chat_model: str, **kwargs) -> Dict:
        """Complete RAG pipeline with RSE"""
        with st.spinner("Calculating chunk relevance..."):
            chunk_values = self.calculate_chunk_values(query, chunks, vector_store, embedding_model)
        
        with st.spinner("Finding optimal segments..."):
            best_segments, scores = self.find_best_segments(chunk_values, **kwargs)
        
        segments = self.reconstruct_segments(chunks, best_segments)
        context = self.format_segments_for_context(segments)
        
        with st.spinner("Generating response..."):
            response = self.generate_response(query, context, chat_model)
        
        return {
            "query": query,
            "segments": segments,
            "response": response,
            "segment_scores": scores
        }
    
    def standard_top_k_retrieval(self, chunks: List[str], vector_store: EnhancedVectorStore, 
                                query: str, embedding_model: str, chat_model: str, k: int = 10) -> Dict:
        """Standard top-k retrieval"""
        with st.spinner("Retrieving relevant chunks..."):
            query_embedding = self.ollama_client.generate_embeddings([query], embedding_model)[0]
            results = vector_store.search(query_embedding, top_k=k)
            retrieved_chunks = [result["document"] for result in results]
        
        context = "\n\n".join([
            f"CHUNK {i+1}:\n{chunk}" 
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        with st.spinner("Generating response..."):
            response = self.generate_response(query, context, chat_model)
        
        return {
            "query": query,
            "chunks": retrieved_chunks,
            "response": response
        }

# Streamlit App
def main():
    st.title("📚 Enhanced RAG System with Ollama")
    st.markdown("Upload a PDF and ask questions using local AI models via Ollama")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedRAGSystem()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Ollama settings
        st.subheader("Ollama Settings")
        ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
        
        # Update client if URL changed
        if ollama_url != st.session_state.get('ollama_url', ''):
            st.session_state.rag_system.ollama_client = OllamaClient(ollama_url)
            st.session_state.ollama_url = ollama_url
        
        # Get available models
        available_models = st.session_state.rag_system.ollama_client.get_available_models()
        
        # Model selection
        embedding_models = [m for m in available_models if 'embed' in m.lower()]
        if not embedding_models:
            embedding_models = ["nomic-embed-text"]
        
        chat_models = [m for m in available_models if 'embed' not in m.lower()]
        if not chat_models:
            chat_models = ["llama3.1", "mistral"]
        
        embedding_model = st.selectbox("Embedding Model", embedding_models)
        chat_model = st.selectbox("Chat Model", chat_models)
        
        # Processing parameters
        st.subheader("Processing Parameters")
        chunk_size = st.slider("Chunk Size", 400, 1200, 800)
        
        # RSE parameters
        st.subheader("RSE Parameters")
        irrelevant_penalty = st.slider("Irrelevant Chunk Penalty", 0.1, 0.5, 0.2)
        max_segment_length = st.slider("Max Segment Length", 10, 30, 20)
        total_max_length = st.slider("Total Max Length", 20, 50, 30)
        min_segment_value = st.slider("Min Segment Value", 0.1, 0.5, 0.2)
        
        # Top-k parameter
        top_k = st.slider("Top-K for Standard Retrieval", 5, 20, 10)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Process document
            if st.button("Process Document") or 'processed_doc' not in st.session_state:
                chunks, vector_store, doc_info = st.session_state.rag_system.process_document(
                    uploaded_file, chunk_size, embedding_model
                )
                
                if chunks is not None:
                    st.session_state.processed_doc = {
                        'chunks': chunks,
                        'vector_store': vector_store,
                        'doc_info': doc_info
                    }
                    st.success("Document processed successfully!")
                    
                    # Show document info
                    st.info(f"""
                    **Document Info:**
                    - Total text length: {doc_info['total_text_length']:,} characters
                    - Number of chunks: {doc_info['chunk_count']}
                    - Average chunk size: {doc_info['total_text_length'] // doc_info['chunk_count']} characters
                    """)
    
    with col2:
        st.header("Query Interface")
        
        if 'processed_doc' in st.session_state:
            # Query input
            query = st.text_area("Enter your question:", height=100)
            
            # Method selection
            method = st.radio("Choose retrieval method:", 
                            ["Both (Comparison)", "RSE Only", "Standard Top-K Only"])
            
            if st.button("Generate Answer") and query:
                chunks = st.session_state.processed_doc['chunks']
                vector_store = st.session_state.processed_doc['vector_store']
                
                if method == "Both (Comparison)":
                    col_rse, col_standard = st.columns(2)
                    
                    with col_rse:
                        st.subheader("🎯 RSE Method")
                        rse_result = st.session_state.rag_system.rag_with_rse(
                            chunks, vector_store, query, embedding_model, chat_model,
                            irrelevant_chunk_penalty=irrelevant_penalty,
                            max_segment_length=max_segment_length,
                            total_max_length=total_max_length,
                            min_segment_value=min_segment_value
                        )
                        
                        st.write("**Answer:**")
                        st.write(rse_result['response'])
                        
                        with st.expander("View Segments"):
                            for i, segment in enumerate(rse_result['segments']):
                                st.write(f"**Segment {i+1}** (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):")
                                st.write(segment['text'][:200] + "..." if len(segment['text']) > 200 else segment['text'])
                                st.write("---")
                    
                    with col_standard:
                        st.subheader("📝 Standard Top-K")
                        standard_result = st.session_state.rag_system.standard_top_k_retrieval(
                            chunks, vector_store, query, embedding_model, chat_model, k=top_k
                        )
                        
                        st.write("**Answer:**")
                        st.write(standard_result['response'])
                        
                        with st.expander("View Retrieved Chunks"):
                            for i, chunk in enumerate(standard_result['chunks']):
                                st.write(f"**Chunk {i+1}:**")
                                st.write(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                                st.write("---")
                
                elif method == "RSE Only":
                    st.subheader("🎯 RSE Method Results")
                    result = st.session_state.rag_system.rag_with_rse(
                        chunks, vector_store, query, embedding_model, chat_model,
                        irrelevant_chunk_penalty=irrelevant_penalty,
                        max_segment_length=max_segment_length,
                        total_max_length=total_max_length,
                        min_segment_value=min_segment_value
                    )
                    
                    st.write("**Answer:**")
                    st.write(result['response'])
                    
                    with st.expander("View Segments and Details"):
                        st.write(f"Found {len(result['segments'])} optimal segments:")
                        for i, segment in enumerate(result['segments']):
                            st.write(f"**Segment {i+1}** (Score: {result['segment_scores'][i]:.4f})")
                            st.write(f"Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}")
                            st.write(segment['text'])
                            st.write("---")
                
                else:  # Standard Top-K Only
                    st.subheader("📝 Standard Top-K Results")
                    result = st.session_state.rag_system.standard_top_k_retrieval(
                        chunks, vector_store, query, embedding_model, chat_model, k=top_k
                    )
                    
                    st.write("**Answer:**")
                    st.write(result['response'])
                    
                    with st.expander("View Retrieved Chunks"):
                        for i, chunk in enumerate(result['chunks']):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(chunk)
                            st.write("---")
        else:
            st.info("Please upload and process a PDF document first.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Make sure Ollama is running locally with the selected models downloaded.")

if __name__ == "__main__":
    main()
