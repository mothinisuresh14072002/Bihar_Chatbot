"""
Build the ChromaDB vector index from extracted text files.
Uses sentence-aware chunking, cosine similarity, and rich metadata.
"""
import sys
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    RAW_DATA_DIR, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, HF_TOKEN
)

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


# ─────────── Sentence-Aware Chunking ───────────

def split_into_sentences(text):
    """Split text into sentences at period, question mark, exclamation,
    or newline boundaries — but not at abbreviations like 'Dr.' or 'Sr.'"""
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    # Filter empty parts
    return [p.strip() for p in parts if p.strip()]


def chunk_by_sentences(text, max_chars=1000, overlap_chars=200):
    """
    Chunk text by sentence boundaries — never breaks mid-sentence.
    Each chunk is max_chars long, with overlap_chars of shared text.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # If adding this sentence exceeds limit, save current chunk
        if current_len + sent_len > max_chars and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            # Overlap: keep last N characters worth of sentences
            overlap_text = ''
            overlap_sents = []
            for s in reversed(current_chunk):
                if len(overlap_text) + len(s) > overlap_chars:
                    break
                overlap_sents.insert(0, s)
                overlap_text = ' '.join(overlap_sents)

            current_chunk = overlap_sents
            current_len = len(overlap_text)

        current_chunk.append(sentence)
        current_len += sent_len + 1  # +1 for space

    # Last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks


# ─────────── Metadata Extraction ───────────

def detect_topic(text):
    """Detect the topic of a chunk based on keywords."""
    text_lower = text.lower()
    topic_keywords = {
        "registration": ["registration", "register", "पंजीकरण", "enroll", "signup"],
        "schemes": ["scheme", "योजना", "benefit", "लाभ", "assistance", "सहायता", "pension", "scholarship"],
        "documents": ["document", "दस्तावेज", "aadhaar", "आधार", "certificate", "proof", "passbook"],
        "approval": ["approval", "approve", "अनुमोदन", "verify", "verification", "sanction"],
        "payment": ["payment", "भुगतान", "amount", "disbursement", "transfer", "राशि"],
        "grievance": ["grievance", "शिकायत", "complaint", "redressal"],
        "login": ["login", "password", "OTP", "लॉगिन", "authentication"],
        "mobile_app": ["mobile app", "android", "ios", "मोबाइल"],
        "cess": ["cess", "उपकर", "collection", "assessment"],
        "employer": ["employer", "नियोक्ता", "establishment", "company"],
        "worker": ["worker", "labour", "labor", "कर्मकार", "श्रमिक", "construction"],
        "renewal": ["renewal", "renew", "नवीनीकरण", "expiry"],
        "ecard": ["ecard", "e-card", "ई-कार्ड", "card", "download"],
        "contact": ["contact", "helpline", "phone", "email", "address", "संपर्क"],
    }
    
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return topic
    return "general"


# ─────────── Main Indexing ───────────

def load_structured_json(json_path):
    """Load structured JSON from PDF extraction (has headings + metadata)."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("  BOCW Knowledge Base — Index Builder")
    print("=" * 60)

    # Load embedding model
    print("\n  📦 Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        token=HF_TOKEN,
    )
    print(f"  ✅ Model loaded: {EMBEDDING_MODEL_NAME}")

    # Initialize ChromaDB with cosine similarity
    print("  🗄️  Initializing ChromaDB (cosine similarity)...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Delete existing collection to rebuild from scratch
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"  🗑️  Deleted old collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass

    # Create with cosine distance for better similarity matching
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # COSINE similarity
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []
    chunk_counter = 0

    # ── Process structured JSON files (from PDF extraction) ──
    json_files = list(RAW_DATA_DIR.glob("*.json"))
    for json_path in json_files:
        print(f"\n  📖 Processing structured: {json_path.name}")
        try:
            sections = load_structured_json(json_path)
            for sec in sections:
                heading = sec.get("heading", "")
                content = sec.get("content", "")
                source = sec.get("source", json_path.stem)
                page = sec.get("page", 0)

                # Prepend heading to content for context
                full_text = f"{heading}. {content}" if heading else content
                
                chunks = chunk_by_sentences(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                
                for i, chunk in enumerate(chunks):
                    topic = detect_topic(chunk)
                    chunk_id = f"{json_path.stem}_s{chunk_counter}"
                    chunk_counter += 1

                    all_chunks.append(chunk)
                    all_ids.append(chunk_id)
                    all_metadatas.append({
                        "source": source,
                        "section": heading,
                        "page": page,
                        "topic": topic,
                        "chunk_index": i,
                        "type": "pdf"
                    })
            
            print(f"  ✅ {len(sections)} sections → chunks created")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # ── Process plain text files (from web scraping, fallback) ──
    txt_files = list(RAW_DATA_DIR.glob("*.txt"))
    for txt_path in txt_files:
        print(f"\n  📄 Processing text: {txt_path.name}")
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            if len(text.strip()) < 50:
                print(f"  ⚠️  Skipping (too short)")
                continue

            chunks = chunk_by_sentences(text, CHUNK_SIZE, CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):
                topic = detect_topic(chunk)
                chunk_id = f"{txt_path.stem}_t{chunk_counter}"
                chunk_counter += 1

                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "source": txt_path.name,
                    "section": "",
                    "page": 0,
                    "topic": topic,
                    "chunk_index": i,
                    "type": "web" if not txt_path.name.startswith("pdf_") else "pdf"
                })

            print(f"  ✅ {len(chunks)} chunks created")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    if not all_chunks:
        print("\n  ❌ No chunks to index! Run extract_pdfs.py first.")
        return

    # ── Embed and store ──
    print(f"\n  🧮 Embedding {len(all_chunks)} chunks...")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True).tolist()

    # ChromaDB batch limit is ~5000, add in batches
    batch_size = 500
    for start in range(0, len(all_chunks), batch_size):
        end = min(start + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[start:end],
            documents=all_chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=all_metadatas[start:end]
        )
    
    print(f"\n  ✅ Indexed {collection.count()} chunks into ChromaDB")

    # ── Topic distribution ──
    topic_counts = {}
    for m in all_metadatas:
        t = m["topic"]
        topic_counts[t] = topic_counts.get(t, 0) + 1
    
    print(f"\n  📊 Topic Distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"     {topic:20s} : {count} chunks")

    # ── Test queries ──
    print(f"\n  🧪 Running test queries...\n")
    test_questions = [
        "What are the BOCW welfare schemes?",
        "How to register as a worker?",
        "What documents are required?",
        "How to track application status?",
        "What is the approval process?",
    ]

    for q in test_questions:
        q_embedding = embedding_model.encode(q).tolist()
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=2
        )
        print(f"  Q: {q}")
        if results['documents'] and results['documents'][0]:
            top_doc = results['documents'][0][0]
            top_score = results['distances'][0][0] if results['distances'] else "?"
            top_topic = results['metadatas'][0][0].get('topic', '?')
            preview = top_doc[:120].replace('\n', ' ') + "..."
            print(f"  → [{top_topic}] (score: {top_score:.3f}) {preview}")
        print()

    print("=" * 60)
    print(f"  ✅ Index build complete!")
    print(f"     Collection: {CHROMA_COLLECTION_NAME}")
    print(f"     Total chunks: {collection.count()}")
    print(f"     Path: {CHROMA_DB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
