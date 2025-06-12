# Graph RAG Alternatives with Local LLM Support

## Overview of Alternatives

Here are the top Graph RAG alternatives that support local LLMs, ranked by ease of use and features:

| Framework | Type | Local LLM Support | Key Strengths | Best For |
|-----------|------|-------------------|---------------|----------|
| **LightRAG** | Graph RAG | ✅ Ollama, HF | Fast, lightweight, simple setup | Quick prototypes, cost-effective |
| **LlamaIndex** | Comprehensive RAG | ✅ Ollama, HF, local | Mature ecosystem, extensive docs | Production applications |
| **LangChain** | RAG Framework | ✅ Ollama, HF, local | Large community, many integrations | Complex workflows |
| **GraphRAG-Local-UI** | Enhanced MS GraphRAG | ✅ Ollama focus | Full UI, based on MS GraphRAG | Users who like MS GraphRAG |
| **Haystack** | AI Orchestration | ✅ Ollama, local | Production-ready, enterprise | Large-scale deployments |
| **RAGFlow** | Visual RAG | ✅ Local models | GUI-based, knowledge graphs | Non-technical users |

---

## 1. LightRAG (Recommended for Local LLMs)

LightRAG is a streamlined approach to retrieval-augmented generation that focuses on simplicity and performance, offering a lightweight implementation that delivers faster and more efficient RAG capabilities compared to more complex alternatives.

### Installation and Setup

```bash
# Install LightRAG
pip install lightrag-hku

# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Basic LightRAG with Ollama Implementation

```python
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# Set working directory
WORKING_DIR = "./lightrag_workdir"

# Initialize LightRAG with Ollama models
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name='llama3.1:8b',  # Your Ollama model
    llm_model_kwargs={"options": {"num_ctx": 32768}},  # 32k context
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, 
            embed_model="nomic-embed-text"
        )
    ),
)

# Insert documents (text files)
with open("your_document.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Query the graph
print("Global Search:")
print(rag.query("What are the main themes?", param=QueryParam(mode="global")))

print("Local Search:")
print(rag.query("Tell me about specific entities", param=QueryParam(mode="local")))

print("Hybrid Search:")
print(rag.query("Detailed analysis", param=QueryParam(mode="hybrid")))
```

### LightRAG with Streamlit Visualization

```python
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
```

---

## 2. LlamaIndex with Local LLMs

LlamaIndex is a comprehensive data framework designed to connect LLMs with private data sources, making it a powerful foundation for building RAG applications.

### Installation

```bash
pip install llama-index
pip install llama-index-llms-ollama
pip install llama-index-embeddings-ollama
```

### LlamaIndex with Ollama Setup

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configure Ollama LLM and embeddings
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Query
response = query_engine.query("What are the main topics in the documents?")
print(response)
```

### LlamaIndex Knowledge Graph with Local LLMs

```python
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore

# Set up graph store
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Create knowledge graph index
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    include_embeddings=True,
)

# Query the knowledge graph
kg_query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

response = kg_query_engine.query("What are the relationships between entities?")
```

---

## 3. GraphRAG-Local-UI (Enhanced Microsoft GraphRAG)

GraphRAG using Local LLMs - Features robust API and multiple apps for Indexing/Prompt Tuning/Query/Chat/Visualizing/Etc. This is meant to be the ultimate GraphRAG/KG local LLM app.

### Installation

```bash
git clone https://github.com/severian42/GraphRAG-Local-UI
cd GraphRAG-Local-UI
pip install -r requirements.txt

# Install Ollama models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Configuration

```yaml
# settings.yaml
llm:
  api_key: "ollama"
  type: openai_chat
  model: llama3.1:8b
  api_base: http://localhost:11434/v1
  model_supports_json: true

embeddings:
  llm:
    api_key: "ollama"
    type: openai_embedding
    model: nomic-embed-text
    api_base: http://localhost:11434/v1
```

### Running the UI

```bash
# Start the main Gradio app
python app.py

# Or start the indexing UI
python index_app.py

# Or start the API server
python api.py
```

---

## 4. LangChain with Local Graph RAG

### Installation

```bash
pip install langchain langchain-ollama
pip install neo4j  # For graph database
```

### LangChain Graph RAG Implementation

```python
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# Initialize Ollama models
llm = OllamaLLM(model="llama3.1:8b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load and process documents
loader = TextLoader("your_document.txt")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Query
result = qa_chain("What are the main themes in the document?")
print(result["result"])
```

---

## 5. Local LLM Options Comparison

### Ollama (Recommended for beginners)
```bash
# Easy installation
curl -fsSL https://ollama.com/install.sh | sh

# Available models for Graph RAG
ollama pull llama3.1:8b        # Good balance of performance/speed
ollama pull llama3.1:70b       # Higher quality, slower
ollama pull mistral:7b         # Fast, efficient
ollama pull phi3:medium        # Small, fast
ollama pull gemma2:9b          # Google's model

# Embedding models
ollama pull nomic-embed-text   # Best for RAG
ollama pull mxbai-embed-large  # Alternative embedding
```

### Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load local model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# For embeddings
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
```

### LlamaCpp (For GGUF models)
```python
from llama_cpp import Llama

# Load GGUF model
llm = Llama(
    model_path="./models/llama-2-7b-chat.gguf",
    n_ctx=4096,  # Context window
    n_gpu_layers=35,  # GPU acceleration
)
```

---

## 6. Performance and Cost Comparison

| Model Size | Speed | Quality | Memory Usage | Best For |
|------------|--------|---------|--------------|----------|
| **7B models** | Fast | Good | 8-16GB RAM | Development, testing |
| **13B models** | Medium | Better | 16-32GB RAM | Balanced use cases |
| **70B models** | Slow | Excellent | 64GB+ RAM | High-quality results |

### Resource Requirements

- **CPU Only**: 7B models work well
- **GPU (8GB)**: 7B-13B models with partial GPU offloading
- **GPU (16GB+)**: 13B-70B models with full GPU acceleration
- **GPU (24GB+)**: 70B models comfortably

---

## 7. Complete Local RAG Pipeline Example

```python
# Complete pipeline combining multiple approaches
import os
from pathlib import Path

class LocalGraphRAG:
    def __init__(self, framework="lightrag", model_size="7b"):
        self.framework = framework
        self.model_size = model_size
        self.setup_models()
    
    def setup_models(self):
        if self.framework == "lightrag":
            self.setup_lightrag()
        elif self.framework == "llamaindex":
            self.setup_llamaindex()
        elif self.framework == "langchain":
            self.setup_langchain()
    
    def process_documents(self, input_dir):
        """Process various file types"""
        # Use the unified input processor from previous artifact
        from process_inputs import process_all_inputs
        process_all_inputs(input_dir, "processed")
    
    def create_knowledge_graph(self):
        """Build the knowledge graph"""
        # Implementation depends on chosen framework
        pass
    
    def query(self, question, mode="global"):
        """Query the knowledge graph"""
        # Framework-specific querying
        pass
    
    def visualize(self):
        """Create Streamlit visualization"""
        # Use the Streamlit app from previous artifact
        pass

# Usage
rag = LocalGraphRAG(framework="lightrag", model_size="7b")
rag.process_documents("./input_docs")
rag.create_knowledge_graph()
result = rag.query("What are the main themes?")
```

---

## 8. Advantages of Local LLMs vs OpenAI

### **Benefits of Local LLMs:**
- **Privacy**: Data never leaves your machine
- **Cost**: No API fees after initial setup
- **Offline**: Works without internet connection
- **Customization**: Fine-tune models for your specific domain
- **Compliance**: Meets strict data governance requirements

### **Trade-offs:**
- **Setup Complexity**: Requires more technical knowledge
- **Hardware Requirements**: Need sufficient RAM/GPU
- **Model Quality**: May be lower than GPT-4 for complex tasks
- **Maintenance**: Need to manage models and updates

### **Best Local Models for Graph RAG (2025):**
1. **Llama 3.1 (8B/70B)**: Best overall performance
2. **Mistral 7B**: Fast and efficient
3. **Phi-3 Medium**: Small but capable
4. **Gemma 2 (9B)**: Google's efficient model

The choice depends on your specific needs: LightRAG for simplicity and speed, LlamaIndex for production applications, or GraphRAG-Local-UI if you prefer the Microsoft approach with local models.