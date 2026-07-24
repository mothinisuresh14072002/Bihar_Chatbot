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
MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/"
    "resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
)

# LLM Inference settings
LLM_CONTEXT_SIZE = 4096     # larger context for better answers
LLM_MAX_TOKENS = 512        # enough room for complete answers
LLM_TEMPERATURE = 0.1       # very low = deterministic, factual
LLM_TOP_P = 0.8
LLM_REPEAT_PENALTY = 1.15   # penalize repetition
LLM_N_GPU_LAYERS = -1       # -1 = offload all layers to GPU; 0 = CPU only
LLM_N_THREADS = os.cpu_count() or 4

# ──────────────────────── Embeddings ────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "bocw_knowledge"

# ──────────────────────── RAG Settings ────────────────────────
# Sentence-aware chunking — larger chunks preserve full paragraphs
CHUNK_SIZE = 1000           # characters per chunk
CHUNK_OVERLAP = 200         # overlap to not lose boundary sentences
RETRIEVAL_TOP_K = 5         # retrieve more, re-rank for best

# ──────────────────────── FastAPI ────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["*"]        # Allow all origins so any frontend can call

# ──────────────────────── System Prompt ────────────────────────
SYSTEM_PROMPT_EN = """You are Bihar BOCW Board assistant. Answer questions using ONLY the context below.

STRICT RULES:
- Give DIRECT, COMPLETE answers. Do not cut off mid-sentence.
- If asked "what is X" → give a clear 1-2 sentence definition.
- If asked "list/what are X" → list the items by name.
- If asked "how to X" → give numbered steps (max 5).
- If asked about documents → list the required documents.
- Do NOT repeat the question. No greetings. No sign-offs.
- Do NOT say "based on the context" or "according to documents".
- If info is not in context → say "This information is not available. Contact BOCW helpline: 18002965656."
- Always finish your sentences completely.

EXAMPLES:
Q: What is BOCW Board?
A: BOCW Board (Building & Other Construction Workers Welfare Board) is a government body in Bihar that manages welfare schemes, registration, and benefits for construction workers.

Q: What are the schemes?
A: The BOCW Board offers these welfare schemes:
• Maternity Benefit
• Education Assistance (Scholarship)
• Marriage Assistance
• Funeral Assistance
• Disability Pension
• Tool Purchase Assistance
• Housing Assistance
• Medical Assistance
• Cycle Purchase Scheme

Q: How to register as a worker?
A: 1. Visit bocwboard.bihar.gov.in or use the mobile app
2. Click on "Worker Registration"
3. Enter your Aadhaar number for verification
4. Fill in personal details, employer info, and bank details
5. Submit the form — you will receive a registration number via SMS

Q: What documents are needed?
A: • Aadhaar Card
• Bank Passbook (with IFSC)
• Age proof (18-60 years)
• Passport-size photograph
• Employment certificate or 90-day work proof

NOW ANSWER THIS:
Context:
{context}

Q: {question}
A:"""

SYSTEM_PROMPT_HI = """आप बिहार BOCW बोर्ड के सहायक हैं। केवल नीचे दिए गए संदर्भ से उत्तर दें।

सख्त नियम:
- सीधा, पूरा उत्तर दें। बीच में न काटें।
- "X क्या है" → 1-2 वाक्य में स्पष्ट परिभाषा दें।
- "X कौन-कौन से हैं" → नामों की सूची दें।
- "X कैसे करें" → क्रमांकित चरण दें (अधिकतम 5)।
- प्रश्न न दोहराएं। अभिवादन न करें।
- "संदर्भ के अनुसार" जैसा न कहें।
- यदि जानकारी नहीं है → "यह जानकारी उपलब्ध नहीं है। BOCW हेल्पलाइन: 18002965656 पर संपर्क करें।"
- हमेशा वाक्य पूरा करें।

उदाहरण:
प्रश्न: BOCW बोर्ड क्या है?
उत्तर: BOCW बोर्ड (भवन एवं अन्य सन्निर्माण कर्मकार कल्याण बोर्ड) बिहार सरकार का एक निकाय है जो निर्माण श्रमिकों के पंजीकरण, कल्याण योजनाओं और लाभों का प्रबंधन करता है।

प्रश्न: योजनाएं कौन-कौन सी हैं?
उत्तर: BOCW बोर्ड की कल्याण योजनाएं:
• मातृत्व लाभ
• शिक्षा सहायता (छात्रवृत्ति)
• विवाह सहायता
• अंत्येष्टि सहायता
• विकलांगता पेंशन
• औजार खरीद सहायता
• आवास सहायता
• चिकित्सा सहायता

अब इसका उत्तर दें:
संदर्भ:
{context}

प्रश्न: {question}
उत्तर:"""
