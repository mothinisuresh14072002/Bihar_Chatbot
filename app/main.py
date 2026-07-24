import logging
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import traceback

from app import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title='BOCW Bihar Chatbot API')

# CORS setup - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_engine = None
index_ready = False

@app.on_event("startup")
async def startup_event():
    global rag_engine, index_ready
    logger.info("Initializing RAG Engine...")
    try:
        from app.rag import RAGEngine
        rag_engine = RAGEngine()
        index_ready = True
        logger.info("RAG Engine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        logger.error(traceback.format_exc())
        rag_engine = None
        index_ready = False

# Ensure static directory exists
config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
index_html_path = config.STATIC_DIR / "index.html"
if not index_html_path.exists():
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>BOCW Bihar Chatbot Testing UI</h1><p>Test the API at /api/chat</p></body></html>")

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

@app.get("/")
async def read_index():
    """Serve the root index.html."""
    return FileResponse(str(config.STATIC_DIR / "index.html"))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = False
    if rag_engine and getattr(rag_engine, "llm", None) is not None:
        model_loaded = True
        
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "index_ready": index_ready
    }

class ChatRequestJSON(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.post("/api/chat")
async def chat_endpoint(request: Request, body: Optional[ChatRequestJSON] = None):
    """
    Chat endpoint that accepts both JSON and Form data.
    """
    content_type = request.headers.get("content-type", "")
    message = None
    session_id = None
    
    try:
        if "application/json" in content_type:
            data = await request.json()
            message = data.get("message")
            session_id = data.get("session_id")
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form_data = await request.form()
            message = form_data.get("message")
            session_id = form_data.get("session_id")
        else:
            raise HTTPException(status_code=415, detail="Unsupported Media Type. Use JSON or Form data.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")
        
    if not message or not str(message).strip():
        raise HTTPException(status_code=400, detail="Message field is required and cannot be empty.")
        
    message = str(message).strip()
    logger.info(f"Query received: {message} | Session ID: {session_id}")
    
    if not index_ready or not rag_engine:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG Engine is not ready. Please ensure the knowledge base is indexed and ChromaDB is populated."}
        )
        
    try:
        response = rag_engine.query(message)
        return response
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred while processing your request."}
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
