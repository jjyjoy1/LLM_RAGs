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

