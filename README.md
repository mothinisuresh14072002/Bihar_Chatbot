# Bihar BOCW RAG Chatbot 🏗️🤖

A fully local **Retrieval-Augmented Generation (RAG)** chatbot for the **Bihar Building & Other Construction Workers Welfare Board (BOCW)**. Answers questions about schemes, registration, benefits, and more — in both **English** and **Hindi**.

## Features

- 🔍 **RAG-powered answers** from BOCW website data + official PDF documents
- 🌐 **Bilingual** — responds in English or Hindi based on user's language
- 🏠 **Fully local** — no cloud APIs, runs on your machine
- 🚀 **Fast inference** — Qwen2.5-3B-Instruct (text-only, quantized)
- 🔌 **API service** — FastAPI backend on port 8000, any frontend can connect

## Quick Start

### One-Click (Windows)
```bash
start.bat
```

### Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the LLM model (~2 GB)
python scripts/download_model.py

# 4. Extract PDF data
python scripts/extract_pdfs.py

# 5. (Optional) Scrape website — requires Playwright
playwright install chromium
python scripts/scrape_website.py

# 6. Build the search index
python scripts/build_index.py

# 7. Start the server
python -m app.main
```

The backend listens on **http://localhost:8000**. Use `/api/chat` from your frontend or client.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat` | Send a question |
| `POST` | `/api/voice-chat` | Upload speech and receive transcript, answer, and WAV reply |

### Chat API

```bash
POST /api/chat
Content-Type: application/json

{
  "message": "What are the BOCW schemes?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "The BOCW Board Bihar provides several schemes...",
  "sources": [{"text": "...", "source": "pdf_srs_bocw.txt", "score": 0.45}],
  "language": "en"
}
```

### Voice API

Send an audio file as multipart form data using the field name `audio`:

```bash
curl -X POST http://localhost:8000/api/voice-chat \
  -F "audio=@question.wav"
```

The response contains `transcript`, the normal RAG answer, and `audio_base64`. Decode
`audio_base64` as an `audio/wav` file to play the voice reply. The first voice request
downloads and loads the Whisper model.

Whisper uses the multilingual `base` model because it supports Hindi and English while
remaining practical for CPU inference. `pyttsx3` provides local text-to-speech using the
server's installed voice engine; install a Hindi system voice if Hindi speech output is
required.

## Architecture

```
┌─────────────────────┐     ┌──────────────┐
│  Website Scraper    │────►│              │
│  (Playwright)       │     │  ChromaDB    │
├─────────────────────┤     │  Vector      │──► RAG Engine ──► FastAPI (8000)
│  PDF Extractor      │────►│  Store       │       │
│  (PyMuPDF)          │     │              │       ▼
└─────────────────────┘     └──────────────┘   Qwen2.5-3B
                                               (llama-cpp)
```

## Project Structure

```
Bihar_Chatbot/
├── app/
│   ├── config.py          # Configuration (paths, model, RAG settings)
│   ├── rag.py             # RAG engine (retrieve + generate)
│   └── main.py            # FastAPI server (port 8000)
├── scripts/
│   ├── scrape_website.py  # Playwright website scraper
│   ├── extract_pdfs.py    # PDF text extractor
│   ├── build_index.py     # ChromaDB indexer
│   └── download_model.py  # Model downloader
├── models/                # GGUF model files
├── data/
│   ├── raw/               # Extracted text files
│   └── chroma_db/         # Persisted vector store
├── requirements.txt
├── start.bat              # Windows one-click launcher
└── README.md
```

## Data Sources

| Source | Content |
|--------|---------|
| `SRS_BOCW v2.4.pdf` | System Requirements — all modules, workflows |
| `User Manual on State BOCW DLC.pdf` | Step-by-step user guide |
| `bocwboard.bihar.gov.in` | Home, schemes, news, tenders, orders, contact |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen2.5-3B-Instruct (GGUF, Q4_K_M) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Backend | FastAPI + uvicorn |
| Scraping | Playwright (headless Chromium) |
| PDF | PyMuPDF |
| Frontend | Vanilla HTML/CSS/JS |

## Helpline

**BOCW Bihar Toll-Free:** 18002965656
