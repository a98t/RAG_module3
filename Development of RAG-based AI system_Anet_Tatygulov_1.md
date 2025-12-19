# Development of RAG-based AI system – Premier League Insight Assistant

**Author:** Anet Tatygulov  
**Date:** December 8, 2025  
**Video Demo:** [TODO: Add link to demo video after recording]

---

## Abstract

The **Premier League Insight Assistant** is a RAG-based AI system designed to answer questions about the English Premier League using a custom knowledge base. The system combines semantic vector search with large language model generation to provide accurate, context-grounded answers about EPL history, analytics, tactics, and fan culture. Built for football enthusiasts, sports analytics students, and anyone interested in learning about the Premier League, this assistant demonstrates practical application of retrieval-augmented generation technology to create a domain-specific question-answering system.

---

## 1. Main Idea

### Problem Statement
Large Language Models often provide generic or outdated information when asked about specific domains. They may hallucinate facts or miss recent developments. The Premier League Insight Assistant solves this by grounding AI responses in a curated knowledge base.

### Solution Overview
When a user asks a question like *"Explain Leicester's miracle season in simple terms"*, the system:

1. **Retrieves** relevant documents from the Premier League knowledge base (e.g., "Leicester's Miracle Season")
2. **Embeds** the question as a vector and searches for semantically similar content
3. **Generates** an answer using an LLM with the retrieved documents as context

This approach ensures:
- **Accuracy**: Answers are based on verified facts from the knowledge base
- **Transparency**: Retrieved documents are shown to the user
- **Relevance**: Semantic search finds the most appropriate context
- **No hallucinations**: LLM is instructed to answer only from provided context

### Target Users
- **Football fans** seeking detailed EPL information
- **Sports analytics students** learning about modern football metrics
- **Researchers** studying tactical evolution and fan culture
- **Content creators** needing accurate Premier League facts

---

## 2. Key Concepts

### 2.1 Retrieval-Augmented Generation (RAG)

RAG is a technique that enhances LLM responses by combining two powerful capabilities:

1. **Retrieval**: Finding relevant information from a knowledge base using semantic search
2. **Generation**: Using an LLM to synthesize that information into coherent, natural language answers

Unlike traditional chatbots that rely solely on pre-trained knowledge, RAG systems dynamically fetch current, domain-specific information before generating responses. This significantly reduces hallucinations and allows the system to provide accurate answers about specialized topics.

**Benefits of RAG:**
- Grounds responses in factual data
- Allows updating knowledge without retraining the model
- Provides source attribution for answers
- Reduces computational costs compared to fine-tuning

### 2.2 Vector Embeddings and Semantic Search

**Embeddings** are numerical representations of text that capture semantic meaning. Each piece of text is converted into a high-dimensional vector (typically 256-1536 dimensions).

**Key principle:** Texts with similar meanings produce similar vectors, even if the exact words differ.

**Example:**
```
"The Premier League generates billions in revenue"
    → [0.234, -0.567, 0.891, ..., 0.123]  (1536 numbers)

"EPL earns massive amounts from broadcasting"
    → [0.241, -0.573, 0.885, ..., 0.119]  (very similar vector!)
```

**Semantic search** uses these embeddings to find relevant documents by:
1. Converting the user query to a vector
2. Calculating similarity (e.g., cosine similarity) between query vector and document vectors
3. Returning documents with highest similarity scores

This is much more powerful than keyword matching—it understands *meaning*, not just words.

### 2.3 Domain Topics Covered

The knowledge base covers five comprehensive aspects of the English Premier League:

1. **Origins & Structure** (5 documents)
   - Formation in 1992, promotion/relegation system
   - Points system, European qualification
   - Financial model and revenue distribution

2. **Iconic Moments** (5 documents)
   - Aguero's title-winning goal (2012)
   - Arsenal's Invincibles season (2003-04)
   - Leicester City's miracle (2015-16)
   - Manchester City's centurion season (2017-18)

3. **Football Analytics** (5 documents)
   - Expected goals (xG) metrics
   - Pressing intensity (PPDA)
   - Heatmaps and player positioning
   - Set-piece analysis, wearable tracking data

4. **Tactics & Playing Styles** (5 documents)
   - 4-3-3 formation and wide play
   - Low blocks and counter-attacks
   - False nines and fluid forwards
   - Build-up from the back, press-resistant midfielders

5. **Fan Culture & Stadium Atmosphere** (5 documents)
   - Home advantage and atmosphere
   - Club anthems and chants
   - Matchday rituals and derby rivalries
   - Global fanbase and social media communities

---

## 3. Dataset Concept

### 3.1 Dataset Structure

The knowledge base consists of **25 carefully curated documents** providing factual information about the Premier League. Each document is a self-contained knowledge chunk with:

- **Title**: Descriptive name for the topic
- **Topic**: One of 5 main categories
- **Tags**: Keywords for additional categorization
- **Content**: 3-5 sentences of dense, factual information

### 3.2 Dataset Characteristics

**Size**: 25 documents (~6,000 words total)

**Rationale for Small Size:**
- Representative coverage of key EPL aspects
- Demonstrates RAG capabilities without overwhelming complexity
- Easy to validate retrieval accuracy
- Quick ingestion and experimentation

**Quality over Quantity:**
- Each document contains unique, specific details (names, numbers, dates)
- Facts are verifiable from official sources
- Information density is high—no filler content
- Documents are distinct enough to test retrieval precision

### 3.3 Annotation Strategy

Each document includes structured metadata:

```json
{
  "id": "doc_006",
  "title": "Aguerooooo Title Winner",
  "topic": "Iconic Moments",
  "tags": ["Manchester City", "title race", "2011-12", "Sergio Aguero", "QPR"],
  "content": "In the 2011–12 season, Manchester City won their first Premier League title..."
}
```

**Metadata purposes:**
- **ID**: Unique identifier for tracking and updates
- **Topic**: Enables category filtering in searches
- **Tags**: Supports keyword-based enhancements to vector search
- **Content**: The actual text that gets embedded and retrieved

### 3.4 Data Sources

Documents are manually curated from:
- Official Premier League historical records
- Sports analytics publications
- Football tactical analysis resources
- Fan culture studies and journalism

All facts include specific details (e.g., "90 points and +47 goal difference") to enable verification that retrieval is working correctly—if the system answers with precise numbers, it must have retrieved the right document.

---

## 4. Design & Architecture

### 4.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                │
│                           ↓                                 │
│              ┌────────────────────────┐                     │
│              │   UI Layer             │                     │
│              │  (Streamlit / CLI)     │                     │
│              └────────────┬───────────┘                     │
│                           ↓                                 │
│              ┌────────────────────────┐                     │
│              │   RAG Pipeline         │                     │
│              │   (rag_pipeline.py)    │                     │
│              └─┬──────────────────┬───┘                     │
│                │                  │                         │
│         ┌──────▼──────┐    ┌─────▼──────┐                  │
│         │ Embeddings  │    │    LLM     │                  │
│         │   Client    │    │   Client   │                  │
│         └──────┬──────┘    └─────▲──────┘                  │
│                │                  │                         │
│                ↓                  │                         │
│         ┌────────────────┐        │                         │
│         │  Vector DB     │        │                         │
│         │  (Weaviate)    │        │                         │
│         │                │        │                         │
│         │  - Stores docs │        │                         │
│         │  - Vector      │        │                         │
│         │    search      │        │                         │
│         └────────────────┘        │                         │
│                │                  │                         │
│                └──────────────────┘                         │
│            Retrieved Context                                │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Description

#### UI Layer (Streamlit or CLI)
- **Responsibility**: Accept user questions, display answers and retrieved context
- **Technologies**: Streamlit for web interface, or Python CLI for terminal interaction
- **Files**: `app.py` (Streamlit) or `app_cli.py` (CLI)

#### RAG Pipeline (`rag_pipeline.py`)
- **Responsibility**: Orchestrate the retrieval-augmented generation flow
- **Process**:
  1. Receive user question
  2. Convert question to embedding vector
  3. Search vector database for top-k similar documents
  4. Build prompt combining retrieved docs + user question
  5. Send prompt to LLM
  6. Return answer and source documents to UI

#### Embeddings Client (`embeddings_client.py`)
- **Responsibility**: Convert text to vector embeddings
- **Model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Methods**:
  - `embed_text(text: str) -> list[float]`: Single text to vector
  - `embed_batch(texts: list[str]) -> list[list[float]]`: Batch processing

#### LLM Client (`llm_client.py`)
- **Responsibility**: Generate natural language responses
- **Model**: OpenAI `gpt-4o-mini` (cost-effective, fast)
- **Configuration**:
  - System prompt: "You are an expert on the Premier League..."
  - Temperature: 0 (deterministic, factual responses)
  - Max tokens: 512 (reasonable answer length)

#### Vector Database (Weaviate)
- **Responsibility**: Store document embeddings and perform similarity search
- **Deployment**: Docker container on localhost:8080
- **Schema**:
  - Class: `PremierLeagueDoc`
  - Properties: `title`, `topic`, `tags`, `content`
  - Vector: 1536-dimensional embedding
  - Index: HNSW with cosine similarity

#### Database Client (`db_client.py`)
- **Responsibility**: Interface with Weaviate
- **Methods**:
  - `create_schema()`: Define database schema
  - `insert_document()`: Add single document with vector
  - `search_similar_docs()`: Vector similarity search
  - `delete_all()`: Clear database (for re-ingestion)

#### Ingestion Script (`scripts/ingest_data.py`)
- **Responsibility**: Load dataset into vector database
- **Process**:
  1. Read documents from `data/premier_league_documents.jsonl`
  2. Generate embeddings for each document
  3. Insert documents + embeddings into Weaviate
  4. Verify successful ingestion

---

## 5. System Technical Details

### 5.1 Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Core development language |
| **Vector DB** | Weaviate | 1.27+ | Document storage & vector search |
| **Embeddings** | OpenAI API | Latest | Text-to-vector conversion |
| **LLM** | OpenAI API | Latest | Answer generation |
| **UI Framework** | Streamlit | 1.28+ | Web interface |
| **Orchestration** | LangChain | 0.1+ | Optional: Chain management |
| **Container** | Docker Desktop | Latest | Weaviate deployment |
| **Environment** | python-dotenv | 1.0+ | Configuration management |

### 5.2 Python Libraries

**Core Dependencies:**
```
openai>=1.0.0
weaviate-client>=4.0.0
streamlit>=1.28.0
python-dotenv>=1.0.0
```

**Optional/Development:**
```
langchain>=0.1.0
langchain-openai>=0.0.5
pandas>=2.0.0
jupyter>=1.0.0
```

### 5.3 API Models and Configuration

#### Embeddings Model
- **Model**: `text-embedding-3-small`
- **Dimensions**: 1536
- **Cost**: ~$0.02 per 1M tokens
- **Speed**: ~1000 texts/second
- **Rationale**: Cost-effective, high-quality embeddings suitable for semantic search

#### LLM Model
- **Model**: `gpt-4o-mini`
- **Context Window**: 128k tokens
- **Cost**: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- **Temperature**: 0 (deterministic)
- **Max Tokens**: 512
- **Rationale**: Fast, affordable, excellent for factual Q&A with context

### 5.4 Weaviate Configuration

**Docker Run Command:**
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=20 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e CLUSTER_HOSTNAME=node1 \
  semitechnologies/weaviate:1.27.0
```

**Schema Definition:**
```python
{
    "class": "PremierLeagueDoc",
    "vectorizer": "none",  # We provide vectors manually
    "vectorIndexType": "hnsw",
    "vectorIndexConfig": {
        "distance": "cosine",
        "efConstruction": 128,
        "maxConnections": 64
    },
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "topic", "dataType": ["text"]},
        {"name": "tags", "dataType": ["text[]"]},
        {"name": "content", "dataType": ["text"]}
    ]
}
```

### 5.5 Project Structure

```
Module 3/
├── data/
│   └── premier_league_documents.jsonl      # Dataset (25 docs)
├── src/
│   ├── __init__.py
│   ├── embeddings_client.py                # Embedding generation
│   ├── llm_client.py                       # LLM interaction
│   ├── db_client.py                        # Weaviate interface
│   └── rag_pipeline.py                     # Core RAG logic
├── scripts/
│   └── ingest_data.py                      # Database ingestion
├── app.py                                  # Streamlit UI
├── app_cli.py                              # CLI alternative
├── requirements.txt                        # Python dependencies
├── .env.example                            # Environment template
├── .env                                    # API keys (gitignored)
├── README.md                               # Setup instructions
├── simple-rag-example.ipynb                # Educational notebook
└── Development of RAG-based AI system_Anet_Tatygulov.md  # This file
```

---

## 6. Requirements

### 6.1 System Requirements

**Hardware:**
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 5GB free space (for Docker, models cache, and data)
- **Network**: Stable internet connection for API calls

**Operating System:**
- **Windows 10/11** with WSL2 (for Docker)
- **macOS** 10.15+ (Catalina or later)
- **Linux** (Ubuntu 20.04+, Debian, Fedora)

### 6.2 Software Requirements

**Required:**
- **Python**: 3.10 or higher
- **Docker Desktop**: Latest stable version
- **pip**: Python package manager (included with Python)
- **Git**: For cloning repository (optional)

**Optional:**
- **VS Code**: Recommended IDE with Python extension
- **Jupyter**: For exploring the example notebook
- **curl/Postman**: For testing API endpoints

### 6.3 API Keys and Access

**OpenAI API Account:**
- Sign up at [platform.openai.com](https://platform.openai.com)
- Add payment method (pay-as-you-go)
- Generate API key from dashboard
- **Estimated cost**: ~$0.10-0.50 for full testing session

**Environment Variables:**
```env
OPENAI_API_KEY=sk-...your-key-here...
```

### 6.4 Python Dependencies

See `requirements.txt` for complete list. Key packages:
- `openai`: Official OpenAI Python client
- `weaviate-client`: Weaviate database SDK
- `streamlit`: Web UI framework
- `python-dotenv`: Environment variable management

---

## 7. Limitations

### 7.1 Knowledge Base Limitations

**Scope Constraints:**
- Dataset contains only 25 documents covering 5 topics
- No information about current season or live matches
- Limited depth on individual players, clubs, or managers
- Focuses on general concepts rather than exhaustive statistics

**Temporal Limitations:**
- Knowledge cutoff: Information up to 2018-2020 era
- No real-time data integration
- Historical events only, no predictive capabilities

**Coverage Gaps:**
- No tactical video analysis
- No financial details beyond general concepts
- No coverage of lower English football divisions
- No player transfer market information

### 7.2 Technical Limitations

**API Dependencies:**
- Requires active internet connection
- Subject to OpenAI API rate limits (3,500 requests/min for tier 1)
- API costs accumulate with usage
- Service availability depends on OpenAI uptime

**Retrieval Quality:**
- Top-k search (k=5) may miss relevant docs if poorly phrased query
- Embedding model may not capture very specific jargon
- No query expansion or reformulation built-in
- Cosine similarity threshold not dynamically adjusted

**Generation Quality:**
- LLM may refuse to answer if context is insufficient
- Potential for over-reliance on single retrieved document
- Temperature=0 reduces creativity but ensures consistency
- 512 token limit may truncate comprehensive answers

### 7.3 Scalability Limitations

**Database:**
- Current setup: Single Docker container on localhost
- Not production-ready (no authentication, no backup)
- No horizontal scaling or replication
- Limited to ~10,000 documents without performance tuning

**Concurrency:**
- Streamlit runs single-threaded by default
- No request queuing or load balancing
- Simultaneous users would create separate API calls (cost multiplication)

### 7.4 User Experience Limitations

**Interface:**
- Basic UI with minimal error handling
- No conversation history or multi-turn dialogue
- No user authentication or personalization
- Retrieved documents shown raw (not formatted nicely)

**Accessibility:**
- No mobile optimization
- No screen reader support
- English language only
- Requires technical setup (not SaaS-ready)

### 7.5 Future Improvements

**Short-term:**
- Add query expansion to improve retrieval
- Implement hybrid search (vector + keyword)
- Add document re-ranking step
- Create conversation memory for follow-up questions

**Medium-term:**
- Expand dataset to 100+ documents
- Add web scraping for recent EPL news
- Implement user feedback loop for answer quality
- Deploy to cloud (e.g., Streamlit Community Cloud)

**Long-term:**
- Multi-modal support (images, video clips)
- Fine-tuned embedding model for football domain
- Integration with live match APIs
- Multi-language support

---

## 8. How to Run

### Quick Start

1. **Clone and Setup:**
   ```bash
   cd "c:\Users\anet.tatygulov\Desktop\EPAM_Course\Module 3"
   pip install -r requirements.txt
   ```

2. **Configure API Key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Start Weaviate:**
   ```bash
   docker run -d --name weaviate -p 8080:8080 \
     -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
     semitechnologies/weaviate:1.27.0
   ```

4. **Ingest Data:**
   ```bash
   python scripts/ingest_data.py
   ```

5. **Run UI:**
   ```bash
   streamlit run app.py
   ```

6. **Ask Questions!**
   - Open browser to `http://localhost:8501`
   - Enter question: "What was Leicester City's miracle season?"
   - View answer with source documents

**Detailed instructions:** See `README.md` file

---

## 9. Video Demonstration

**Video Link:** [TODO: Add YouTube/Google Drive link after recording]

**Video Contents (1-3 minutes):**

1. **Repository Overview (15 sec)**
   - Show project structure in VS Code
   - Highlight key files: dataset, source code, UI

2. **Dataset Preview (15 sec)**
   - Open `premier_league_documents.jsonl`
   - Scroll through a few entries showing structure

3. **Docker and Database (20 sec)**
   - Show Docker Desktop with Weaviate container running
   - Terminal: `docker ps` to verify status

4. **Data Ingestion (20 sec)**
   - Run `python scripts/ingest_data.py`
   - Show successful ingestion logs

5. **Streamlit UI Demo (60 sec)**
   - Start app: `streamlit run app.py`
   - Ask question: "Tell me about the Invincibles"
   - Show retrieved context documents
   - Show generated answer based on context
   - Ask second question: "What is xG in football analytics?"
   - Demonstrate different topic retrieval

6. **Closing (10 sec)**
   - Quick summary of RAG workflow
   - Show all components working together

---

## 10. Conclusion

The **Premier League Insight Assistant** successfully demonstrates a complete RAG-based AI system implementation. By combining semantic vector search with large language model generation, the system provides accurate, context-grounded answers about the English Premier League.

**Key Achievements:**
- ✅ Custom dataset with 25 annotated documents across 5 topics
- ✅ Vector database (Weaviate) deployed and operational
- ✅ Embeddings pipeline for semantic search
- ✅ LLM integration for natural language generation
- ✅ Functional web UI (Streamlit) and CLI alternative
- ✅ Complete RAG pipeline connecting all components
- ✅ Reproducible setup with clear documentation

**Educational Value:**
This project serves as a practical template for building domain-specific RAG systems, demonstrating:
- How to structure and annotate knowledge bases
- Vector embedding generation and storage
- Semantic search implementation
- Prompt engineering for RAG scenarios
- End-to-end system integration

The architecture is modular and extensible, allowing easy adaptation to other domains by simply replacing the dataset and adjusting prompts.

---

## Appendix: References and Resources

**Technologies:**
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/chat-completions)
- [Streamlit Documentation](https://docs.streamlit.io)

**RAG Concepts:**
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)

**Premier League Data Sources:**
- Official Premier League website
- Sports analytics publications (StatsBomb, Opta)
- Football tactical analysis resources

---

**Document Version:** 1.0  
**Last Updated:** December 8, 2025  
**Contact:** [Your email/GitHub]
