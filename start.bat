@echo off
echo ============================================================
echo   Bihar BOCW RAG Chatbot - Launcher
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Check if venv exists
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo [SETUP] Virtual environment created.
)

:: Activate venv
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo [SETUP] Installing dependencies...
pip install -r requirements.txt --quiet

:: Check if model exists
if not exist "models\qwen2.5-3b-instruct-q4_k_m.gguf" (
    echo.
    echo [MODEL] Downloading Qwen2.5-3B-Instruct model...
    python scripts\download_model.py
)

:: Check if ChromaDB index exists
if not exist "data\chroma_db" (
    echo.
    echo [INDEX] No knowledge base found. Building...
    echo [INDEX] Step 1: Extracting PDFs...
    python scripts\extract_pdfs.py
    echo [INDEX] Step 2: Building search index...
    python scripts\build_index.py
    echo.
    echo [NOTE] To also scrape the BOCW website, run:
    echo        python scripts\scrape_website.py
    echo        python scripts\build_index.py
)

echo.
echo ============================================================
echo   Starting BOCW Chatbot Server on http://localhost:8000
echo ============================================================
echo.
echo   API:      http://localhost:8000/api/chat
echo   Health:   http://localhost:8000/api/health
echo.
echo   Press Ctrl+C to stop the server.
echo ============================================================
echo.

python -m app.main

pause
