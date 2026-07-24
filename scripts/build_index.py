import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    RAW_DATA_DIR, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME, 
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
)

import chromadb
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def main():
    print("Loading embedding model...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return

    print("Initializing ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        # Clear existing collection
        try:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
            print(f"Deleted existing collection '{CHROMA_COLLECTION_NAME}'")
        except Exception:
            pass # Collection didn't exist
            
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    all_files = list(RAW_DATA_DIR.glob("*.txt"))
    if not all_files:
        print(f"No text files found in {RAW_DATA_DIR}")
        return

    total_chunks = 0
    total_docs = len(all_files)
    
    for file_path in all_files:
        print(f"Processing {file_path.name}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                continue
                
            embeddings = embedding_model.encode(chunks).tolist()
            
            ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]
            
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_chunks += len(chunks)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    print(f"\nIndexing complete!")
    print(f"Total documents processed: {total_docs}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Collection size: {collection.count()}")

    # Test Query
    test_query = "What are the BOCW schemes?"
    print(f"\nRunning test query: '{test_query}'")
    try:
        query_embedding = embedding_model.encode([test_query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        for i, doc in enumerate(results['documents'][0]):
            print(f"\nResult {i+1} (Source: {results['metadatas'][0][i]['source']}):\n{doc[:200]}...")
    except Exception as e:
        print(f"Error running test query: {e}")

if __name__ == "__main__":
    main()
