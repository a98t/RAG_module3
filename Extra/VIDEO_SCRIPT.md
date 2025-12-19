# Video Script - Premier League Insight Assistant Demo
**Duration:** 90 seconds

---

## 🎬 Script (Word-for-word)

### Opening (0:00-0:10)
*[Show folder structure in VS Code]*

"Hello! I'm presenting my RAG-based AI system - the Premier League Insight Assistant. Let me show you how it works from data to deployment."

---

### Step 1: Dataset (0:10-0:20)
*[Open data/premier_league_documents.jsonl]*

"First, my dataset: 22 annotated documents covering five Premier League topics - origins, iconic moments, analytics, tactics, and fan culture. Each document has a unique ID, title, topic, tags, and detailed content."

---

### Step 2: Vector Database (0:20-0:30)
*[Run: docker ps]*

"For storage, I'm using Weaviate vector database running in Docker on localhost port 8080. It provides HNSW indexing with cosine similarity search."

---

### Step 3: Code Architecture (0:30-0:45)
*[Show src/ folder briefly]*

"The system has four main components: embeddings client using OpenAI's text-embedding-3-small with 1536 dimensions, LLM client using GPT-4o-mini, database client for Weaviate operations, and the RAG pipeline that orchestrates everything."

---

### Step 4: Data Ingestion (0:45-0:55)
*[Run: python scripts/ingest_data.py and show output]*

"The ingestion script loads documents, generates embeddings in batch, and inserts them into Weaviate. Done - 22 documents indexed and searchable."

---

### Step 5: Live Demo (0:55-1:20)
*[Open browser at localhost:8501, type question]*

"Now the UI - built with Streamlit. Let me ask: 'Explain the 4-3-3 formation.'"

*[Show results appearing]*

"The system converts my question to a vector, searches Weaviate, retrieves the top 5 relevant documents - here's 'The 4-3-3 and Wide Wingers' with 69% similarity - passes them to the LLM with my question, and generates this comprehensive answer with sources."

---

### Step 6: Another Example (1:20-1:25)
*[Ask another question quickly]*

"Let me try another: 'What is xG in football analytics?' Same process - instant retrieval and context-aware answer."

---

### Closing (1:25-1:30)
*[Show files briefly]*

"All code is on GitHub with documentation, requirements, and setup instructions. Thank you!"

---

## 📋 Quick Tips for Recording

1. **Before recording:**
   - Have browser open at http://localhost:8501
   - Have VS Code ready with files visible
   - Have terminal ready for commands
   - Close unnecessary tabs/windows

2. **Screen setup:**
   - Use 1920x1080 resolution
   - Zoom VS Code text to 150% for visibility
   - Use browser zoom 125% for Streamlit UI

3. **Practice these questions:**
   - "Explain the 4-3-3 formation"
   - "What is xG in football analytics?"
   - "Tell me about Leicester's 2015-16 season"

4. **Voice recording:**
   - Speak clearly and at moderate pace
   - Pause 1 second between sections
   - Show enthusiasm but stay professional

5. **Recording tools:**
   - **Windows:** Use Xbox Game Bar (Win+G) - built-in, free
   - **Alternative:** OBS Studio - free, more features
   - **Quick option:** Loom.com - instant sharing

6. **File management:**
   - Save as MP4 format
   - Upload to YouTube (unlisted) or Google Drive
   - Copy link and paste in your .md file

---

## ⏱️ Timing Breakdown

| Section | Time | What to Show |
|---------|------|--------------|
| Intro | 0:00-0:10 | Folder structure |
| Dataset | 0:10-0:20 | JSONL file content |
| Docker | 0:20-0:30 | `docker ps` output |
| Code | 0:30-0:45 | src/ folder files |
| Ingestion | 0:45-0:55 | Script output |
| Demo 1 | 0:55-1:20 | 4-3-3 formation Q&A |
| Demo 2 | 1:20-1:25 | xG question |
| Closing | 1:25-1:30 | Final thoughts |

**Total: 90 seconds**
