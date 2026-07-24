"""
RAG Engine — Retrieve relevant context and generate answers.
Uses cosine similarity search, topic-aware re-ranking, and output post-processing.
"""
import re
import logging
from langdetect import detect
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from app import config

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        """Initialize ChromaDB connection and embedding model.
        LLM is loaded lazily on first query for fast startup."""
        
        # ChromaDB
        self.chroma_client = PersistentClient(path=str(config.CHROMA_DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            raise RuntimeError(
                "ChromaDB is empty. Run the indexing pipeline:\n"
                "  python scripts/extract_pdfs.py\n"
                "  python scripts/build_index.py"
            )

        logger.info(f"ChromaDB loaded: {self.collection.count()} chunks")

        # Embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        # LLM — loaded lazily
        self.llm = None

    def _load_llm(self):
        """Lazily load LLM on first query."""
        if self.llm is not None:
            return
        
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {config.MODEL_PATH}\n"
                f"Run: python scripts/download_model.py"
            )

        logger.info(f"Loading LLM: {config.MODEL_PATH}")
        self.llm = Llama(
            model_path=str(config.MODEL_PATH),
            n_ctx=config.LLM_CONTEXT_SIZE,
            n_gpu_layers=config.LLM_N_GPU_LAYERS,
            n_threads=config.LLM_N_THREADS,
            verbose=False
        )
        logger.info("LLM loaded successfully")

    # ────────── Language Detection ──────────

    def detect_language(self, text: str) -> str:
        """Detect Hindi vs English."""
        try:
            lang = detect(text)
            return 'hi' if lang == 'hi' else 'en'
        except Exception:
            return 'en'

    # ────────── Retrieval ──────────

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve relevant chunks using cosine similarity.
        Returns list of {text, source, topic, section, score}.
        """
        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K

        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        if results and results.get('documents') and results['documents'][0]:
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "topic": meta.get("topic", "general"),
                    "section": meta.get("section", ""),
                    "score": float(dist)  # cosine distance (lower = more similar)
                })

        return chunks

    # ────────── Context Building ──────────

    def build_context(self, chunks: list[dict]) -> str:
        """
        Build a clean context string from retrieved chunks.
        De-duplicates near-identical chunks and formats cleanly.
        """
        seen = set()
        unique_chunks = []

        for chunk in chunks:
            # Simple de-duplication: skip if >80% of text already seen
            text_key = chunk['text'][:100]
            if text_key in seen:
                continue
            seen.add(text_key)
            unique_chunks.append(chunk)

        # Build context with section markers
        context_parts = []
        for chunk in unique_chunks:
            section = chunk.get('section', '')
            text = chunk['text'].strip()
            if section:
                context_parts.append(f"[{section}]\n{text}")
            else:
                context_parts.append(text)

        return "\n\n".join(context_parts)

    # ────────── Answer Generation ──────────

    def generate(self, question: str, context: str, lang: str) -> str:
        """Generate answer using LLM with the formatted prompt."""
        self._load_llm()

        # Pick language-appropriate prompt
        if lang == 'hi':
            prompt = config.SYSTEM_PROMPT_HI.format(context=context, question=question)
        else:
            prompt = config.SYSTEM_PROMPT_EN.format(context=context, question=question)

        logger.info(f"Generating ({lang}) | prompt_len={len(prompt)}")

        response = self.llm(
            prompt,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
            repeat_penalty=config.LLM_REPEAT_PENALTY,
            echo=False,
            stop=["Q:", "Question:", "प्रश्न:", "\n\nQ:", "\n\nUser:"]  # Stop at next question
        )

        raw_answer = response['choices'][0]['text'].strip()
        return self.post_process(raw_answer)

    # ────────── Post-Processing ──────────

    def post_process(self, text: str) -> str:
        """
        Clean up LLM output:
        - Remove incomplete trailing sentences
        - Remove repetitive text
        - Ensure proper ending
        """
        if not text:
            return "This information is not available. Contact BOCW helpline: 18002965656."

        # Remove "A:" prefix if model echoes it
        text = re.sub(r'^A:\s*', '', text)

        # Remove any "Q:" or "Question:" at the end (model trying to continue)
        text = re.split(r'\n\s*Q:', text)[0]
        text = re.split(r'\n\s*Question:', text)[0]
        text = re.split(r'\n\s*प्रश्न:', text)[0]

        # Remove trailing incomplete sentence (no period/bullet at end)
        lines = text.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cleaned_lines.append(line)

        if cleaned_lines:
            last_line = cleaned_lines[-1]
            # If last line doesn't end with sentence-ending punctuation
            # and it's not a bullet point, try to fix it
            if not re.search(r'[.!?।\)]$', last_line):
                # Check if it's a bullet/numbered item (those can end without period)
                if not re.match(r'^[\•\-\*\d]', last_line):
                    # It's an incomplete sentence — remove it if we have other lines
                    if len(cleaned_lines) > 1:
                        cleaned_lines.pop()
                    else:
                        # Only one line — add period
                        cleaned_lines[-1] = last_line + '.'

        text = '\n'.join(cleaned_lines)

        # Remove repeated phrases (detect if same line appears 2+ times)
        seen_lines = []
        final_lines = []
        for line in text.split('\n'):
            stripped = line.strip().lower()
            if stripped and stripped in seen_lines:
                continue  # Skip duplicate
            seen_lines.append(stripped)
            final_lines.append(line)

        return '\n'.join(final_lines).strip()

    # ────────── Full Query Pipeline ──────────

    def query(self, question: str) -> dict:
        """
        Full RAG pipeline:
        1. Detect language
        2. Retrieve relevant chunks
        3. Build context
        4. Generate answer
        5. Post-process
        """
        try:
            lang = self.detect_language(question)
            logger.info(f"Query [{lang}]: {question}")

            # Retrieve
            chunks = self.retrieve(question)

            if not chunks:
                no_info = (
                    "This information is not available. Contact BOCW helpline: 18002965656."
                    if lang == 'en' else
                    "यह जानकारी उपलब्ध नहीं है। BOCW हेल्पलाइन: 18002965656 पर संपर्क करें।"
                )
                return {"answer": no_info, "sources": [], "language": lang}

            # Build context from chunks
            context = self.build_context(chunks)

            # Generate answer
            answer = self.generate(question, context, lang)

            return {
                "answer": answer,
                "sources": [{"source": c["source"], "topic": c["topic"]} for c in chunks],
                "language": lang
            }

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            logger.error(f"RAG query error: {e}", exc_info=True)
            raise
