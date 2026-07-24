"""
Configuration for Bihar BOCW RAG Chatbot.
All paths, model settings, and RAG parameters in one place.
"""
import os
from pathlib import Path

# ──────────────────────── Base Paths ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, CHROMA_DB_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────── PDF Sources ────────────────────────
PDF_FILES = [
    PROJECT_ROOT / "SRS_BOCW v2.4.pdf",
    PROJECT_ROOT / "User Manual on State BOCW DLC.pdf",
]

# ──────────────────────── Website Scraping ────────────────────────
BOCW_BASE_URL = "https://bocwboard.bihar.gov.in"
SCRAPE_PAGES = [
    {"path": "/", "name": "home", "label": "Home Page"},
    {"path": "/about-us", "name": "about_us", "label": "About Us"},
    {"path": "/our-leader", "name": "leaders", "label": "Our Leaders"},
    {"path": "/order", "name": "orders", "label": "Orders & Circulars"},
    {"path": "/whatsnew", "name": "whats_new", "label": "What's New"},
    {"path": "/tender", "name": "tenders", "label": "Tenders"},
    {"path": "/contact-us", "name": "contact", "label": "Contact Us"},
    {"path": "/worker-registration", "name": "worker_registration", "label": "Worker Registration"},
    {"path": "/employer-registration", "name": "employer_registration", "label": "Employer Registration"},
    {"path": "/scheme", "name": "schemes", "label": "Schemes"},
    {"path": "/faq", "name": "faq", "label": "FAQ"},
    {"path": "/gallery", "name": "gallery", "label": "Gallery"},
    {"path": "/grievance", "name": "grievance", "label": "Grievance"},
    {"path": "/cess-payer-registration", "name": "cess_payer", "label": "Cess Payer Registration"},
]

# ──────────────────────── LLM Model ────────────────────────
# Text-only Qwen2.5-3B-Instruct — fast, excellent Hindi/English
MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/"
    "resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
)

# LLM Inference settings
LLM_CONTEXT_SIZE = 2048
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.85
LLM_N_GPU_LAYERS = -1   # -1 = offload all layers to GPU; 0 = CPU only
LLM_N_THREADS = os.cpu_count() or 4

# ──────────────────────── Embeddings ────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "bocw_knowledge"

# ──────────────────────── RAG Settings ────────────────────────
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 80        # character overlap between chunks
RETRIEVAL_TOP_K = 3       # fewer chunks = more focused context

# ──────────────────────── FastAPI ────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["*"]      # Allow all origins so any frontend can call

# ──────────────────────── System Prompt ────────────────────────
SYSTEM_PROMPT_EN = """You are the Bihar BOCW Board assistant. Follow these rules STRICTLY:

RULES:
1. Answer ONLY the specific question asked — nothing more.
2. Keep answers SHORT: 2-4 bullet points max.
3. Use simple, easy language.
4. Do NOT repeat the question. Do NOT add greetings or sign-offs.
5. Do NOT mention "sources", "context", "documents" or "helpline" unless specifically asked.
6. If asked "what is X", give a 1-2 line definition only.
7. If asked "what are the schemes", list scheme names only — no descriptions.
8. If asked "how to do X", give numbered steps only — max 4 steps.
9. If information is not available, say: "This information is not available. Contact BOCW helpline: 18002965656."
10. Never make up information. Use only the context below.

Context:
{context}

Q: {question}
A:"""

SYSTEM_PROMPT_HI = """आप बिहार BOCW बोर्ड के सहायक हैं। इन नियमों का सख्ती से पालन करें:

नियम:
1. केवल पूछे गए प्रश्न का उत्तर दें — कुछ अतिरिक्त नहीं।
2. उत्तर छोटा रखें: अधिकतम 2-4 बुलेट बिंदु।
3. सरल भाषा का उपयोग करें।
4. प्रश्न न दोहराएं। अभिवादन या अंतिम वाक्य न जोड़ें।
5. जब तक विशेष रूप से न पूछा जाए, "स्रोत", "संदर्भ", "दस्तावेज़" या "हेल्पलाइन" का उल्लेख न करें।
6. यदि "X क्या है" पूछा जाए, तो केवल 1-2 पंक्ति की परिभाषा दें।
7. यदि "योजनाएं क्या हैं" पूछा जाए, तो केवल योजनाओं के नाम बताएं।
8. यदि "X कैसे करें" पूछा जाए, तो केवल क्रमांकित चरण दें — अधिकतम 4 चरण।
9. यदि जानकारी उपलब्ध नहीं है: "यह जानकारी उपलब्ध नहीं है। BOCW हेल्पलाइन: 18002965656 पर संपर्क करें।"
10. कभी भी जानकारी न बनाएं। केवल नीचे दिए गए संदर्भ का उपयोग करें।

संदर्भ:
{context}

प्रश्न: {question}
उत्तर:"""
