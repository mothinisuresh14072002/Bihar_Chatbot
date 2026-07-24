import os
import logging
from langdetect import detect
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from app import config

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        # Load ChromaDB collection from config.CHROMA_DB_DIR
        self.chroma_client = PersistentClient(path=str(config.CHROMA_DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
        
        # If ChromaDB is empty, raise a clear error
        if self.collection.count() == 0:
            raise RuntimeError("ChromaDB is empty. Please run the indexing pipeline first to populate the knowledge base.")
            
        # Load embedding model (sentence-transformers all-MiniLM-L6-v2)
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        # LLM loaded lazily
        self.llm = None
        
    def _load_llm(self):
        """Lazily load the LLM to speed up initialization."""
        if self.llm is None:
            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(f"LLM model file not found at {config.MODEL_PATH}. Please run 'python scripts/download_model.py' to download it.")
            
            logger.info(f"Loading LLM from {config.MODEL_PATH}")
            self.llm = Llama(
                model_path=str(config.MODEL_PATH),
                n_ctx=config.LLM_CONTEXT_SIZE,
                n_gpu_layers=config.LLM_N_GPU_LAYERS,
                n_threads=config.LLM_N_THREADS,
                verbose=False
            )
            
    def detect_language(self, text: str) -> str:
        """Detect if query is Hindi or English."""
        try:
            lang = detect(text)
            return 'hi' if lang == 'hi' else 'en'
        except Exception:
            return 'en' # Default to English on error
            
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Embed the query and search ChromaDB for top_k most similar chunks."""
        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K
            
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        chunks = []
        if results and 'documents' in results and results['documents']:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            # ChromaDB returns distances (lower is closer for L2)
            distances = results['distances'][0] if 'distances' in results and results['distances'] else [0] * len(documents)
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "score": float(dist)
                })
        return chunks
        
    def generate(self, question: str, context_chunks: list[dict]) -> str:
        """Format prompt and generate answer using LLM."""
        self._load_llm()
        lang = self.detect_language(question)
        
        # Format context
        context_str = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in context_chunks])
        
        # Pick correct system prompt
        if lang == 'hi':
            prompt = config.SYSTEM_PROMPT_HI.format(context=context_str, question=question)
        else:
            prompt = config.SYSTEM_PROMPT_EN.format(context=context_str, question=question)
            
        logger.info(f"Generating answer in language: {lang}")
        
        response = self.llm(
            prompt,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
            stop=["User's question:", "Question:", "Context:", "I don't have this information.", "Ensure all provided information"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
        
    def query(self, question: str) -> dict:
        """Full RAG pipeline: retrieve context and generate an answer."""
        try:
            lang = self.detect_language(question)
            chunks = self.retrieve(question)
            
            if not chunks:
                answer = "I don't have this information. Please contact the BOCW helpline at 18002965656." if lang == 'en' else "मेरे पास यह जानकारी नहीं है। कृपया BOCW हेल्पलाइन 18002965656 पर संपर्क करें।"
            else:
                answer = self.generate(question, chunks)
                
            return {
                "answer": answer,
                "sources": chunks,
                "language": lang
            }
        except Exception as e:
            logger.error(f"Error in RAG query pipeline: {str(e)}")
            raise
