I've created a complete Streamlit application that integrates your adaptive RAG system with Ollama. Here are the key changes I made:
Key Modifications
1. Ollama Integration

Replaced OpenAI client with a custom OllamaClient class
Embedding models: Uses local models like nomic-embed-text
Chat models: Uses local models like llama3, mistral, etc.
API calls: Direct HTTP requests to Ollama's REST API

2. Streamlit Interface

File uploader: Select and upload PDF documents
Model selection: Choose from available Ollama models
Configuration panel: Adjust chunk size, overlap, and retrieval parameters
Real-time processing: Process documents and get answers interactively

3. Enhanced Features

Connection checking: Verifies Ollama is running
Model detection: Lists available models automatically
Error handling: Graceful error messages and fallbacks
Session state: Maintains processed documents across interactions
Query examples: Built-in examples for different query types

Setup Instructions
1. Install Ollama
bash# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
2. Pull Required Models
bash# For embeddings
ollama pull nomic-embed-text

# For chat (choose one or more)
ollama pull llama3
ollama pull mistral
ollama pull codellama
3. Install Python Dependencies
bashpip install streamlit numpy PyMuPDF requests pathlib
4. Run the Application
bashstreamlit run adaptive_rag_ollama_streamlit.py
Features
📄 Document Processing:

Upload any PDF document
Configurable chunking parameters
Real-time processing feedback

🤖 Adaptive Querying:

Automatic query classification (Factual/Analytical/Opinion/Contextual)
Strategy-specific retrieval for each query type
Context-aware responses

⚙️ Configuration:

Select chat and embedding models
Adjust retrieval parameters
Optional user context input

📊 Results Display:

Query type identification
Generated answers
Retrieved document snippets with similarity scores

The app will automatically detect your available Ollama models and let you choose which ones to use. Make sure you have at least one embedding model (like nomic-embed-text) and one chat model (like llama3) installed!

