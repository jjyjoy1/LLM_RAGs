# 📚 Novel Knowledge Graph Analyzer

> **AI-Powered Literary Analysis Tool** - Extract characters, relationships, themes, and custom features from novels using local LLMs and interactive knowledge graphs.

## 🌟 Features

### 📖 **Core Functionality**
- **PDF Text Extraction**: Upload and process PDF novels with progress tracking
- **Knowledge Graph Creation**: AI-powered entity and relationship extraction using LlamaIndex
- **Local LLM Support**: Works with Ollama (Llama 3.1, Mistral, Phi-3, etc.) - no API costs!
- **Interactive Visualization**: Dynamic network graphs with Plotly and NetworkX

### 🎯 **Custom Feature Analysis**
- **Define Custom Features**: Create your own categories (e.g., "Magical Objects", "Political Themes")
- **Predefined Templates**: Quick-add common literary elements (emotions, conflicts, relationships)
- **Multi-Feature Analysis**: Analyze individual features or combinations together
- **Cross-Feature Relationships**: Discover connections between different feature types

### 🔍 **Advanced Querying**
- **General Queries**: Ask questions about the entire novel
- **Feature-Specific Queries**: Focus on specific categories (characters, themes, custom features)
- **Predefined Questions**: Ready-to-use literary analysis prompts
- **Multiple Response Modes**: Tree summarization, compact, or simple responses

### 📊 **Analytics & Visualization**
- **Interactive Network Graphs**: Color-coded entities with relationship mapping
- **Feature Relationship Matrix**: Heatmap showing cross-category interactions
- **Entity Analytics**: Distribution charts, connection analysis, and statistics
- **Exportable Results**: Download CSV files for further analysis

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull Required Models**
   ```bash
   ollama pull llama3.1:8b          # Main LLM (8GB RAM recommended)
   ollama pull nomic-embed-text     # Embedding model
   ```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/novel-knowledge-graph-analyzer.git
   cd novel-knowledge-graph-analyzer
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run novel_analyzer.py
   ```

4. **Open in Browser**
   Navigate to `http://localhost:8501`

## 📋 Requirements

```txt
streamlit>=1.28.0
llama-index>=0.9.0
llama-index-llms-ollama>=0.1.0
llama-index-embeddings-ollama>=0.1.0
PyPDF2>=3.0.0
plotly>=5.15.0
networkx>=3.1
pandas>=2.0.0
sentence-transformers>=2.2.0
```

## 🎯 Usage Guide

### 1. Upload & Process Novel
- Upload your PDF novel through the web interface
- Extract text with real-time progress tracking
- Preview extracted content before analysis

### 2. Define Custom Features
Create custom categories to extract specific elements:

```python
# Example custom features
features = {
    "Magical Objects": ["wand", "spell", "potion", "magic", "artifact"],
    "Romantic Elements": ["love", "romance", "kiss", "marriage", "attraction"],
    "Conflict Types": ["battle", "fight", "war", "argument", "rivalry"],
    "Political Themes": ["power", "government", "revolution", "authority"]
}
```

### 3. Create Knowledge Graph
- Generate AI-powered knowledge graph using local LLMs
- Automatic entity extraction and relationship mapping
- Enhanced categorization with custom features

### 4. Explore & Analyze
- **Interactive Networks**: Visualize entity relationships
- **Feature Analysis**: Examine specific categories individually or together
- **Query Interface**: Ask questions about characters, themes, and relationships
- **Export Results**: Download analysis data for academic research

## 🔧 Configuration

### Model Selection

Choose models based on your hardware:

| Model | RAM Required | Speed | Quality | Best For |
|-------|-------------|--------|---------|----------|
| `phi3:medium` | 4-8GB | Fast | Good | Quick analysis, testing |
| `llama3.1:8b` | 8-16GB | Medium | Excellent | Balanced performance |
| `llama3.1:70b` | 64GB+ | Slow | Outstanding | High-quality analysis |
| `mistral:7b` | 8-12GB | Fast | Very Good | Efficient processing |

### Custom Feature Templates

Pre-built templates for different genres:

**Fantasy Novels:**
- Magical Systems, Mythical Creatures, Artifacts, Prophecies

**Historical Fiction:**
- Social Issues, Historical Events, Period Objects, Cultural Elements

**Science Fiction:**
- Technology, Scientific Concepts, Future Society, Space Elements

**Romance Novels:**
- Relationship Types, Emotional States, Social Settings, Romantic Tension

## 📊 Example Analysis

### Input Novel Features
```
📚 "The Lord of the Rings"

Custom Features Defined:
- Characters: Frodo, Gandalf, Aragorn, Legolas...
- Magical Objects: One Ring, Sting, Palantír...
- Locations: Rivendell, Mordor, Gondor...
- Relationships: Fellowship, Romance, Rivalry...
```

### Generated Insights
```
🔍 Query: "How do magical objects influence character relationships?"

📈 Results:
- 47 character entities extracted
- 23 magical objects identified  
- 156 relationships mapped
- 12 cross-feature connections found

💡 Key Insight: "The One Ring creates both alliance and betrayal 
   patterns among characters, serving as both unifying quest 
   objective and source of internal conflict..."
```

## 🎨 Screenshots

### Main Dashboard
*Interactive knowledge graph with custom feature filtering*

### Custom Feature Definition
*Easy-to-use interface for defining literary elements*

### Feature Relationship Analysis
*Heatmap showing interactions between different feature categories*

### Query Interface
![Query Interface](docs/images/query-interface.png)
*Feature-specific and general query capabilities*

## 🏗️ Architecture

```
📁 Project Structure
├── Graph_RAG_LiamaIndex_LLM_app.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── docs/                     # Documentation and examples
│   ├── images/              # Screenshots and diagrams
│   └── examples/            # Example novels and analyses
└── tests/                   # Unit tests (optional)
```

### Technical Stack
- **Frontend**: Streamlit for interactive web interface
- **AI Engine**: LlamaIndex for knowledge graph creation
- **LLM Backend**: Ollama for local model serving
- **Visualization**: Plotly + NetworkX for interactive graphs
- **Data Processing**: Pandas for analysis and export

## 🔬 Advanced Features

### Multi-Dimensional Analysis
```python
# Analyze relationships between multiple feature types
selected_features = ["Characters", "Magical Objects", "Locations"]
cross_relationships = analyzer.get_feature_relationships(feature1, feature2)
```

### Feature Interaction Matrix
```python
# Generate heatmap of feature category interactions
matrix = analyzer.create_feature_relationship_matrix()
```

### Export Capabilities
- **CSV Export**: Entity lists and relationship tables
- **Network Data**: Graph structures for external analysis
- **Query Results**: AI-generated insights and analysis

## 🎓 Use Cases

### Academic Research
- **Literature Analysis**: Character development, thematic exploration
- **Comparative Studies**: Cross-novel relationship patterns
- **Digital Humanities**: Large-scale literary corpus analysis

### Creative Writing
- **Plot Analysis**: Identify relationship gaps and opportunities
- **Character Development**: Visualize character interaction patterns
- **World Building**: Map complex fantasy/sci-fi universes

### Education
- **Teaching Literature**: Interactive exploration of classic novels
- **Student Projects**: Visual analysis of assigned readings
- **Research Training**: Introduction to digital literary analysis

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make Your Changes**
4. **Add Tests** (if applicable)
5. **Submit a Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/novel-knowledge-graph-analyzer.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Start development server
streamlit run Graph_RAG_LiamaIndex_LLM_app.py --server.runOnSave true
```

## 🐛 Troubleshooting

### Common Issues

**Ollama Not Found**
```bash
# Make sure Ollama is running
ollama serve

# Check if models are available
ollama list
```

**Memory Issues**
- Use smaller models (`phi3:medium` instead of `llama3.1:70b`)
- Reduce chunk size in knowledge graph creation
- Process shorter novels for testing

**Slow Processing**
- Ensure GPU acceleration is enabled
- Use faster models (`mistral:7b`)
- Reduce max entities in visualization

**Empty Results**
- Check PDF text extraction quality
- Verify custom feature keywords are relevant
- Try different LLM models

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LlamaIndex Team** for the excellent knowledge graph framework
- **Ollama Project** for making local LLMs accessible
- **Streamlit** for the intuitive web framework
- **NetworkX & Plotly** for powerful visualization capabilities

## 📚 Citations

If you use this tool in academic research, please cite:

```bibtex
@software{novel_knowledge_graph_analyzer,
  title={Novel Knowledge Graph Analyzer: AI-Powered Literary Analysis Tool},
  author={jiyang Jiang},
  year={2025},
  url={https://github.com/jjyjoy1/GraphRAG_LLM}
}
```

## 🔗 Related Projects

- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLM applications
- [Ollama](https://github.com/ollama/ollama) - Local LLM serving platform
- [Streamlit](https://github.com/streamlit/streamlit) - Web app framework for ML/AI

---

