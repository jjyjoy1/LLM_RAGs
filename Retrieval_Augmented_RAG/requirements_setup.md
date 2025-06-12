# Enhanced RAG System Setup Guide

## Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
PyMuPDF>=1.23.0
pytesseract>=0.3.10
Pillow>=10.0.0
numpy>=1.24.0
requests>=2.31.0
```

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for robust PDF extraction)

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**On macOS:**
```bash
brew install tesseract
```

**On Windows:**
- Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to your PATH environment variable

### 3. Install and Setup Ollama

**Install Ollama:**
- Visit https://ollama.ai and download for your platform
- Or use: `curl -fsSL https://ollama.ai/install.sh | sh` (Linux/macOS)

**Download Required Models:**
```bash
# For embeddings (required)
ollama pull nomic-embed-text

# For chat completion (choose one or more)
ollama pull llama3.1
ollama pull mistral
ollama pull llama3.2
ollama pull codellama
```

**Start Ollama Server:**
```bash
ollama serve
```

### 4. Run the Application

```bash
streamlit run enhanced_rag_app.py
```

## Key Features

### 🔧 **Robust PDF Extraction**
- **Primary Method**: PyMuPDF for standard PDF text extraction
- **Fallback OCR**: Automatic OCR using Tesseract for scanned PDFs
- **Text Cleaning**: Advanced preprocessing to fix common PDF extraction issues
- **Multi-method Detection**: Automatically determines if OCR is needed

### 🤖 **Full Ollama Integration**
- **Local Processing**: Everything runs locally - no API keys needed
- **Embedding Models**: Uses `nomic-embed-text` or other compatible models
- **Chat Models**: Supports any Ollama chat model (llama3.1, mistral, etc.)
- **Model Management**: Automatic detection of available models

### 🎯 **Enhanced RAG Methods**
- **RSE (Relevant Segment Extraction)**: Finds coherent text segments
- **Standard Top-K**: Traditional chunk-based retrieval
- **Side-by-side Comparison**: Compare both methods simultaneously

### 🖥️ **Streamlit Interface**
- **File Upload**: Drag-and-drop PDF upload
- **Real-time Processing**: Progress bars and status updates
- **Parameter Control**: Adjust chunk sizes, penalties, and model settings
- **Results Visualization**: Expandable sections to view retrieved content

## Usage Tips

### 📄 **For Best PDF Processing:**
1. **Standard PDFs**: Works great with text-based PDFs
2. **Scanned PDFs**: Automatically uses OCR when needed
3. **Complex Layouts**: The robust extractor handles tables and multi-column layouts
4. **Large Files**: Processes efficiently with progress tracking

### 🔧 **Parameter Tuning:**
- **Chunk Size**: 800 chars works well for most documents
- **Irrelevant Penalty**: Higher values (0.3-0.4) for more focused results
- **Max Segment Length**: Increase for longer coherent passages
- **Top-K**: Start with 10, adjust based on document size

### 🎯 **When to Use Each Method:**
- **RSE**: Better for questions requiring coherent context
- **Standard Top-K**: Faster, good for factual queries
- **Comparison Mode**: Use when evaluating answer quality

## Troubleshooting

### Common Issues:

**1. Ollama Connection Error:**
- Ensure Ollama is running: `ollama serve`
- Check the base URL in the sidebar (default: http://localhost:11434)

**2. Model Not Found:**
- Download required models: `ollama pull nomic-embed-text`
- Refresh the model list in the app

**3. OCR Not Working:**
- Install Tesseract OCR for your platform
- Ensure it's in your PATH

**4. Memory Issues:**
- Reduce chunk size for large documents
- Process fewer chunks at once

**5. Slow Processing:**
- Use smaller models for faster inference
- Reduce the number of chunks/segments processed

## Model Recommendations

### For Embeddings:
- `nomic-embed-text`: Best general-purpose embedding model
- `mxbai-embed-large`: Higher quality, slower

### For Chat:
- `llama3.1`: Best overall performance
- `llama3.2`: Faster, good for simple queries  
- `mistral`: Good alternative, efficient
- `codellama`: Better for technical documents

## Advanced Configuration

### Custom Ollama Setup:
If running Ollama on a different host/port, update the base URL in the sidebar.

### Performance Optimization:
- Use GPU acceleration if available
- Adjust batch sizes in the code for your hardware
- Consider using quantized models for faster inference

## Example Workflow

1. **Start Ollama**: `ollama serve`
2. **Download Models**: `ollama pull nomic-embed-text llama3.1`
3. **Run App**: `streamlit run enhanced_rag_app.py`
4. **Upload PDF**: Use the file uploader
5. **Process Document**: Click "Process Document"
6. **Ask Questions**: Enter queries and select comparison mode
7. **Review Results**: Compare RSE vs Standard methods