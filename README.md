# Hybrid RAG System for Tabular Data

A RAG  system that handles both structured queries (counts, aggregations, lookups) and semantic queries (meaning-based search) on Excel/CSV data.

## Features

- **Hybrid Query Routing**: LLM-based router intelligently directs queries to the appropriate pipeline
- **Text-to-Code**: Pandas agent for structured queries (counts, filters, aggregations)
- **Semantic Search**: Hybrid retrieval combining BM25 + vector embeddings
- **Local Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Real-time Progress**: Background processing with live progress tracking
- **Production Logging**: Timestamped logs with processing metrics

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Django 5.2 |
| Frontend | Streamlit |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers |
| LLM | Groq (Llama 3.3 70B) |
| Text-to-Code | LangChain Pandas Agent |
| Retrieval | BM25 + Semantic (EnsembleRetriever) |

## Project Structure

```
rag_project/
├── backend/
│   ├── settings.py
│   └── urls.py
├── api/
│   ├── engine.py      # Core RAG engine
│   ├── views.py       # API endpoints
│   └── urls.py        # API routing
├── app.py             # Streamlit frontend
├── requirements.txt
├── .env               # API keys 
└── .gitignore
```

## Setup

### 1. Clone and create virtual environment

```bash
cd rag_project
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from: https://console.groq.com/keys

### 4. Run the backend

```bash
python manage.py runserver
```

### 5. Run the frontend (new terminal)

```bash
streamlit run app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload/` | POST | Upload CSV/Excel file |
| `/api/status/` | GET | Check processing status |
| `/api/chat/` | POST | Send query, get response |
| `/api/health/` | GET | Health check |

## Architecture

```
User Query
    |
    v
[LLM Router] --> STRUCTURED --> [Pandas Agent] --> Execute Python --> Result
    |
    +--> SEMANTIC --> [Hybrid Retriever] --> [LLM] --> Answer
                           |
                      BM25 (50%) + Embeddings (50%)
```

### Query Routing

**STRUCTURED** (Text-to-Code):
- Counts and aggregations ("how many", "total")
- Filters ("where status is open")
- Lookups ("ticket 1121")
- Groupby ("breakdown by type")

**SEMANTIC** (RAG):
- Theme-based search ("complaints about shipping")
- Content understanding ("what are customers saying")
- Similarity search ("issues like battery problems")

## Configuration

Key settings in `api/engine.py`:

```python
BATCH_SIZE = 500      # Documents per embedding batch
NUM_WORKERS = 4       # Parallel embedding workers
k = 10                # Top-k documents for retrieval
weights = [0.5, 0.5]  # BM25 vs Semantic balance
```

## Performance

Tested on 8,469 customer support tickets:

| Metric | Value |
|--------|-------|
| Query response | 2-5 seconds |
| Embedding rate | ~14 docs/sec |
| Routing accuracy | 100% |

## Limitations

- CPU-only embeddings (slow for large datasets)
- Row-as-document chunking (not optimal for 100k+ rows)
- No conversation memory (follow-up questions need full context)
- Free tier LLM rate limits

## Production Recommendations

For scaling to 100k+ rows:

| Component | Demo | Production |
|-----------|------|------------|
| Embeddings | Local CPU | OpenAI API / GPU |
| Vector DB | ChromaDB | Milvus / Pinecone |
| SQL | Pandas | PostgreSQL |
| Queue | Threading | Celery + Redis |
| Cache | None | Redis |

## License

MIT
