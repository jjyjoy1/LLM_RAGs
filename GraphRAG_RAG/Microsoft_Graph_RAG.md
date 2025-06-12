# Microsoft Graph RAG Setup Guide with OpenAI and PDFs

## Prerequisites

### 1. Environment Setup
```bash
# Install required packages
pip install graphrag
pip install openai
pip install pypdf2 or pymupdf  # for PDF processing
pip install python-dotenv
```

### 2. API Keys Required
- OpenAI API key (for LLM and embeddings)
- Optional: Azure OpenAI if using Azure services

## Step 1: Project Structure

Create your project directory structure:
```
graph_rag_project/
├── input/
│   ├── pdfs/           # Your PDF files go here
│   └── processed/      # Processed text files
├── output/             # Graph RAG outputs
├── .env               # Environment variables
├── settings.yaml      # Configuration file
└── process_pdfs.py    # PDF processing script
```

## Step 2: Environment Configuration
Create a `.env` file:

## Step 3: PDF Processing Script
Create `process_pdfs.py`:

## Step 4: Graph RAG Configuration
Create `settings.yaml`:

## Step 5: Running the Pipeline
### Initialize Graph RAG
```bash
# Initialize the project
python -m graphrag.index --init --root ./

# This creates default prompt files in prompts/ directory
```

### Process Your Data
```bash
# First, convert PDFs to text
python process_pdfs.py

# Run the indexing pipeline
python -m graphrag.index --root ./
```

### Query the Knowledge Graph
python query_example.py


## Step 6: Advanced Configuration Options

### Custom Prompts
You can customize the extraction prompts in the `prompts/` directory:
- `entity_extraction.txt` - Controls what entities are extracted
- `summarize_descriptions.txt` - How entity descriptions are summarized
- `claim_extraction.txt` - What claims/facts are identified
- `community_report.txt` - How community summaries are generated

### Performance Tuning
- Adjust `chunk.size` and `chunk.overlap` based on your content
- Modify `parallelization.num_threads` based on your system
- Use `gpt-3.5-turbo` instead of `gpt-4` for faster/cheaper processing
- Enable `cache` to avoid reprocessing

### Memory and Cost Optimization
```yaml
# For large document sets, consider:
llm:
  model: gpt-3.5-turbo  # More cost-effective
  max_tokens: 2000      # Reduce for cost savings

chunk:
  size: 200            # Smaller chunks for better granularity
  overlap: 50          # Reduce overlap to save on processing

parallelization:
  num_threads: 10      # Reduce if hitting rate limits
```

## Step 7: Usage Examples

### Local Search (Specific Questions)
```python
result = await search_engine.asearch("What are the key findings about climate change?")
```

### Global Search (Broad Topics)
```python
# For global search, use GlobalSearch instead of LocalSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch

global_search = GlobalSearch(
    llm=llm,
    context_builder=global_context_builder,
    token_encoder=token_encoder,
    max_data_tokens=12000,
)

result = await global_search.asearch("Summarize all major themes across the documents")
```

## Troubleshooting

### Common Issues:
1. **PDF Text Extraction**: Some PDFs may have poor text extraction. Consider using OCR for scanned documents
2. **Rate Limits**: Adjust `stagger` and `num_threads` if hitting OpenAI rate limits
3. **Memory Issues**: Reduce chunk sizes or process documents in batches
4. **Empty Results**: Check that PDF text extraction worked and files are in the correct format

### Monitoring Progress:
- Check `output/artifacts/` for generated files
- Review logs for any processing errors
- Verify text extraction quality from PDF files

This setup will create a knowledge graph from your PDF documents and allow you to perform both local (specific) and global (broad) queries using OpenAI's language models.

## Step 8: Streamlit Dynamic Visualization

### Additional Dependencies
```bash
pip install streamlit
pip install plotly
pip install networkx
pip install pandas
pip install pyvis
pip install streamlit-agraph
```

### Streamlit App Structure
Create `streamlit_app.py`:

### Running the Streamlit App

```bash
# Navigate to your project directory
cd graph_rag_project

# Run the Streamlit app
streamlit run streamlit_app.py
```

### App Features

#### 1. Query Interface
- **Local Search**: Specific, detailed questions about entities and relationships
- **Global Search**: Broad questions requiring synthesis across the entire knowledge base
- Real-time search with context display

#### 2. Interactive Network Graph
- **Dynamic Visualization**: Nodes represent entities, edges show relationships
- **Color Coding**: Different colors for entity types (person, organization, etc.)
- **Size Mapping**: Node size reflects entity importance/degree
- **Hover Details**: Click/hover for entity descriptions and relationships
- **Adjustable Complexity**: Slider to control number of displayed entities

#### 3. Analytics Dashboard
- **Entity Type Distribution**: Pie chart showing breakdown of entity types
- **Degree Distribution**: Histogram of entity connection counts
- **Top Connected Entities**: Bar chart of most influential entities
- **Filterable Entity Table**: Searchable table with entity details

#### 4. Community Analysis
- **Community Size Distribution**: Analysis of community structures
- **Community Reports**: AI-generated summaries of entity clusters
- **Interactive Community Explorer**: Drill down into specific communities

### Advanced Configuration

#### Custom Styling
Create `config.toml` in `.streamlit/` directory:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

#### Performance Optimization
```python
# Add to streamlit_app.py for large datasets
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_large_graph_data():
    # Your data loading logic here
    pass

# Use pagination for large entity lists
def paginate_entities(entities_df, page_size=100):
    total_pages = len(entities_df) // page_size + 1
    page = st.selectbox("Page", range(1, total_pages + 1))
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    return entities_df.iloc[start_idx:end_idx]
```

### Deployment Options

#### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

#### Cloud Deployment
- **Streamlit Cloud**: Connect GitHub repo for automatic deployment
- **Heroku**: Use Procfile with `web: streamlit run streamlit_app.py --server.port $PORT`
- **AWS/Azure**: Deploy using container services

This comprehensive setup creates a powerful, interactive interface for exploring your PDF-derived knowledge graph with real-time querying and dynamic visualizations.
