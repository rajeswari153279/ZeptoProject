# ============================================================
# MODULE 3 - PART A: DOCUMENTS -> CHUNKS -> EMBEDDINGS -> CHROMADB
# ============================================================
import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb

DOCS_FOLDER = "support_assistant/documents"
CHUNK_SIZE = 300  # characters per chunk

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="support_assistant/chroma_db")
collection = client.get_or_create_collection("policies")

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks, current = [], []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= size:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks

all_chunks, all_ids, all_metadata = [], [], []
doc_id = 0

for filepath in glob.glob(f"{DOCS_FOLDER}/*.md"):
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{filename}_{i}")
        all_metadata.append({"source": filename})
    doc_id += 1
    print(f"'{filename}' -> {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

embeddings = model.encode(all_chunks).tolist()

collection.add(
    ids=all_ids,
    embeddings=embeddings,
    documents=all_chunks,
    metadatas=all_metadata,
)
print(f"Stored {len(all_chunks)} chunks in ChromaDB (support_assistant/chroma_db)")

# Quick test retrieval
test_query = "How do I return a damaged item?"
query_embedding = model.encode([test_query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=3)
print(f"\n--- Test query: '{test_query}' ---")
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"[{meta['source']}] (distance={dist:.3f}): {doc[:100]}...")

print("\n--- PART A COMPLETE ---")