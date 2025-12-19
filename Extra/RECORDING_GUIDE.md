# 🎬 Quick Video Recording Checklist

## Before You Start Recording

### ✅ Pre-Recording Checklist

- [ ] Docker is running: `docker ps` shows Weaviate container
- [ ] Streamlit is ready to launch: `streamlit run app.py`
- [ ] Browser ready at http://localhost:8501
- [ ] VS Code open with project folder visible
- [ ] Close all unnecessary windows/tabs
- [ ] Zoom VS Code text to 150% for visibility
- [ ] Test your microphone

---

## 🎤 90-Second Script

### Section 1: Introduction (0:00-0:10)
**Show:** VS Code folder structure  
**Say:** "Hello! I'm presenting my RAG-based AI system - the Premier League Insight Assistant. Let me show you how it works from data to deployment."

### Section 2: Dataset (0:10-0:20)
**Show:** Open `data/premier_league_documents.jsonl`  
**Say:** "First, my dataset: 22 annotated documents covering five Premier League topics - origins, iconic moments, analytics, tactics, and fan culture."

### Section 3: Database (0:20-0:30)
**Show:** Terminal with `docker ps`  
**Say:** "For storage, I'm using Weaviate vector database running in Docker on localhost port 8080."

### Section 4: Architecture (0:30-0:45)
**Show:** `src/` folder files  
**Say:** "The system has four components: embeddings client using OpenAI's text-embedding-3-small, LLM client using GPT-4o-mini, database client, and the RAG pipeline."

### Section 5: Ingestion (0:45-0:55)
**Show:** Run `python scripts/ingest_data.py`  
**Say:** "The ingestion script loads documents, generates embeddings, and inserts them into Weaviate. Done - 22 documents indexed."

### Section 6: Live Demo (0:55-1:20)
**Show:** Browser at localhost:8501, type question  
**Say:** "Now the UI - built with Streamlit. Let me ask: 'Explain the 4-3-3 formation.' The system searches Weaviate, retrieves relevant documents, and generates this comprehensive answer with sources."

### Section 7: Another Example (1:20-1:25)
**Show:** Ask second question  
**Say:** "Let me try: 'What is xG in football analytics?' Same process - instant answer."

### Section 8: Closing (1:25-1:30)
**Show:** Files briefly  
**Say:** "All code is on GitHub with documentation. Thank you!"

---

## 🎥 Recording Steps

### Option 1: Windows Game Bar (Easiest)
1. Press **Win + G**
2. Click "Capture" widget
3. Click record button (or Win + Alt + R)
4. Record your demo
5. Press Win + Alt + R to stop
6. Video saves to `C:\Users\[YourName]\Videos\Captures`

### Option 2: OBS Studio (Professional)
1. Download from obsproject.com
2. Add "Display Capture" source
3. Click "Start Recording"
4. Record demo
5. Click "Stop Recording"
6. Find video in default output folder

### Option 3: Loom (Quick Online)
1. Go to loom.com
2. Click "Get Loom for Free"
3. Install browser extension
4. Click extension → "Start Recording"
5. Instant shareable link when done

---

## 📤 After Recording

### 1. Upload Video

**YouTube (Recommended):**
1. Go to youtube.com → Create → Upload video
2. Select your video file
3. Set visibility to **"Unlisted"**
4. Copy the link

**Google Drive:**
1. Upload to drive.google.com
2. Right-click → Share → Anyone with link can view
3. Copy the link

### 2. Add Link to .md File

Open `Development of RAG-based AI system_Anet_Tatygulov.md`

Change line 6 from:
```markdown
**Video Demo:** [TODO: Add link to demo video after recording]
```

To:
```markdown
**Video Demo:** [Watch Demo](YOUR_LINK_HERE)
```

Save the file.

### 3. Final Check

- [ ] Video link added to .md file
- [ ] Video is accessible (test the link)
- [ ] .md file is properly formatted
- [ ] Filename is correct: `Development of RAG-based AI system_Anet_Tatygulov.md`

---

## 🚀 Submit!

1. Go to your learning platform
2. Find assignment: "Development of RAG-based AI system"
3. Upload: `Development of RAG-based AI system_Anet_Tatygulov.md`
4. Click "Submit"

---

## ✨ Tips for Great Recording

- **Speak clearly** at moderate pace
- **Show enthusiasm** but stay professional
- **Practice once** before recording
- **Don't worry** about minor mistakes
- **Keep it under 90 seconds** - tight and focused
- **Smile** - it shows in your voice!

---

## 🎯 You've Got This!

Your system is **complete and working**. Just show what you built, explain it clearly, and submit. Good luck! 🍀
