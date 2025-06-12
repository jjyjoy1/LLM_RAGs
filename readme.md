# LLM_RAGs: Local LLM with Multiple RAG Algorithms 🤖📚

A comprehensive Streamlit-powered application that integrates local Large Language Models (LLMs) with multiple Retrieval-Augmented Generation (RAG) algorithms, enabling intelligent document interaction without relying on cloud-based AI services.

## 🚀 Features

### 🏠 **Local LLM Integration**
- **Privacy-First**: Run completely locally without sending data to external APIs
- **Cost-Effective**: No per-query costs or API limitations
- **High Performance**: Optimized for local hardware configurations
- **Offline Capability**: Works without internet connectivity

### 🧠 **Multiple RAG Algorithms**
- **Standard RAG**: Traditional semantic search and retrieval
- **Advanced RAG Variants**: 
  - Hierarchical RAG for complex documents
  - Multi-vector RAG for enhanced context understanding
  - Hybrid search combining semantic and keyword matching
  - Re-ranking algorithms for improved relevance

### 🎨 **Streamlit Web Interface**
- **Intuitive UI**: User-friendly chat interface
- **Document Management**: Easy upload and processing of various file formats
- **Real-time Interaction**: Instant responses and dynamic conversations
- **Algorithm Selection**: Switch between different RAG approaches on-the-fly
- **Chat History**: Persistent conversation memory

### 📄 **Document Processing**
- **Multi-Format Support**: PDF, TXT, DOCX, Markdown files
- **Intelligent Chunking**: Optimized text segmentation strategies
- **Vector Embeddings**: Efficient semantic representation
- **Metadata Extraction**: Preserve document structure and context

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM Backend**: Local language models (Ollama/Hugging Face/Custom)
- **Vector Database**: FAISS/ChromaDB/Qdrant
- **Document Processing**: LangChain/LlamaIndex
- **Embeddings**: Local embedding models
- **Python**: 3.8+

## 🚦 Quick Start

### Prerequisites
- Python 3.8 or higher
- Local LLM setup (Ollama recommended)
- Sufficient RAM (8GB+ recommended)
- GPU (optional, for enhanced performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jjyjoy1/LLM_RAGs.git
   cd LLM_RAGs
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up local LLM**
   ```bash
   # Install Ollama (example)
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull mistral  # or your preferred model
   ```

5. **Configure the application**
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your preferred settings
   ```

### Running the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser.

## 📖 Usage Guide

### 1. **Document Upload**
- Click "Browse files" to upload your documents
- Supported formats: PDF, TXT, DOCX, MD
- Wait for processing completion

### 2. **Select RAG Algorithm**
- Choose from available RAG variants in the sidebar
- Each algorithm optimized for different use cases:
  - **Standard RAG**: General-purpose retrieval
  - **Hierarchical RAG**: Best for structured documents
  - **Multi-vector RAG**: Enhanced context understanding
  - **Hybrid Search**: Combines semantic and keyword matching

### 3. **Configure Parameters**
- Adjust chunk size and overlap
- Set similarity thresholds
- Modify retrieval parameters

### 4. **Chat Interface**
- Type your questions in the chat input
- Receive AI-generated responses based on your documents
- View source citations and confidence scores

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  RAG Engine     │────│   Local LLM     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Vector Store   │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Documents     │
                       └─────────────────┘
```

## 🔧 Configuration

### Model Configuration
```yaml
llm:
  model_name: "mistral"
  temperature: 0.1
  max_tokens: 2048
  
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  
rag:
  chunk_size: 1000
  chunk_overlap: 200
  similarity_threshold: 0.7
```

### RAG Algorithm Settings
- **Standard RAG**: Basic semantic similarity search
- **Hierarchical RAG**: Multi-level document understanding
- **Multi-vector RAG**: Multiple embedding representations
- **Hybrid Search**: Semantic + BM25 keyword matching

## 📊 Performance Optimization

### Hardware Recommendations
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB+ for large document sets
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: SSD for faster document processing

### Optimization Tips
- Use GPU acceleration when available
- Adjust chunk sizes based on document types
- Enable caching for frequently accessed documents
- Consider model quantization for memory efficiency

## 🧪 RAG Algorithm Comparison

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| Standard RAG | General documents | Simple, fast | Limited context |
| Hierarchical RAG | Structured docs | Better organization | Complex setup |
| Multi-vector RAG | Rich content | Enhanced accuracy | Higher compute |
| Hybrid Search | Mixed content | Balanced retrieval | Parameter tuning |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## 📋 Roadmap

- [ ] **Enhanced RAG Algorithms**
  - Graph-based RAG
  - Contextual compression
  - Multi-modal RAG (images + text)

- [ ] **UI Improvements**
  - Document visualization
  - Advanced search filters
  - Batch processing interface

- [ ] **Performance Features**
  - Distributed processing
  - Model quantization options
  - Advanced caching strategies

- [ ] **Integration Options**
  - API endpoints
  - Plugin architecture
  - External vector database support

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Model not found"
```bash
# Solution: Ensure your local LLM is properly installed
ollama list  # Check available models
ollama pull <model-name>  # Download if missing
```

**Issue**: "Out of memory error"
```bash
# Solution: Reduce batch size or chunk size in config.yaml
chunk_size: 512  # Reduce from default 1000
batch_size: 4    # Reduce processing batch size
```

**Issue**: "Slow processing"
- Enable GPU acceleration if available
- Reduce document chunk overlap
- Use smaller embedding models
- Consider document preprocessing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for RAG framework
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Ollama](https://ollama.ai/) for local LLM infrastructure
- [Hugging Face](https://huggingface.co/) for model hosting and tools

## 📞 Contact

- **Author**: jjyjoy1
- **Repository**: [https://github.com/jjyjoy1/LLM_RAGs](https://github.com/jjyjoy1/LLM_RAGs)
- **Issues**: [GitHub Issues](https://github.com/jjyjoy1/LLM_RAGs/issues)

---

⭐ **Star this repository if you find it helpful!** ⭐

*Built with ❤️ for the open-source AI community*