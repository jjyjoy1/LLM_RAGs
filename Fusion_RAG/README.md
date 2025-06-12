I've created a comprehensive Streamlit application that integrates your fusion RAG system with Ollama. Here are the key modifications and new features:
Key Changes
1. Ollama Integration

Custom OllamaClient: Handles both embeddings and chat completions
Local models: Uses models like nomic-embed-text for embeddings and llama3 for chat
Error handling: Graceful handling of Ollama connection issues

2. Fusion RAG Features

Vector + BM25 combination: Merges semantic similarity with keyword matching
Configurable fusion weight (α): Adjust the balance between vector and BM25 scores
Score normalization: Ensures fair combination of different scoring methods

3. Advanced Streamlit Interface
Configuration Panel:

Model selection (chat and embedding)
Document processing parameters
Fusion weight slider
Comparison mode toggle

Visualization Features:

Interactive score charts showing vector, BM25, and combined scores
Tabbed interface for comparing different retrieval methods
Real-time processing indicators

Setup Instructions
1. Install Dependencies
bashpip install streamlit numpy PyMuPDF requests pathlib rank-bm25 scikit-learn plotly pandas
2. Install and Setup Ollama
bash# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required models
ollama pull nomic-embed-text  # For embeddings
ollama pull llama3           # For chat
3. Run the Application
bashstreamlit run fusion_rag_ollama_streamlit.py
Key Features
🔀 Fusion Retrieval

Vector retrieval: Semantic similarity using embeddings
BM25 retrieval: Keyword-based matching using TF-IDF
Smart combination: Weighted fusion of both approaches

📊 Comparison Mode
Compare three approaches side-by-side:

Vector-only: Pure semantic search
BM25-only: Pure keyword search
Fusion: Combined approach

🎯 Configuration Options

Fusion weight (α):

0.0 = Pure BM25 (keyword-focused)
0.5 = Equal weight (recommended)
1.0 = Pure vector (semantic-focused)



📈 Interactive Visualizations

Bar charts showing individual and combined scores
Score breakdowns for each retrieved document
Real-time comparison metrics

Why Fusion RAG?
Vector retrieval excels at:

Understanding semantic meaning
Finding conceptually similar content
Handling synonyms and paraphrases

BM25 retrieval excels at:

Exact keyword matches
Technical terms and specific phrases
Named entities and proper nouns

Fusion combines both strengths to provide:

Better recall and precision
Handling of both semantic and literal queries
More robust retrieval across different query types

The app automatically detects your available Ollama models and provides an intuitive interface to experiment with different fusion weights and compare retrieval approaches!

