import os
import chromadb
from sentence_transformers import SentenceTransformer

NOTES_DIR = "data/notes"
CHROMA_PATH = "data/chroma_db"
CHUNK_SIZE = 300

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("notes")

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks, current = [], []
    count = 0
    for word in words:
        current.append(word)
        count += len(word) + 1
        if count >= size:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks

def ingest_notes():
    for filename in os.listdir(NOTES_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(NOTES_DIR, filename)
        with open(filepath, "r") as f:
            text = f.read()
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": filename}] * len(chunks)
        )
        print(f"Ingested {len(chunks)} chunks from {filename}")

if __name__ == "__main__":
    ingest_notes()