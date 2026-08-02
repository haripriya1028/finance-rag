from sentence_transformers import SentenceTransformer
import chromadb
import json

# Load chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print(f"Loaded {len(all_chunks)} chunks")

# Load embedding model (free, runs locally)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Chroma (persistent local database)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="finance_filings")

# Embed and store in batches (faster, avoids memory issues)
batch_size = 100
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i + batch_size]
    texts = [chunk["text"] for chunk in batch]
    ids = [chunk["chunk_id"] for chunk in batch]
    metadatas = [{"company": chunk["company"]} for chunk in batch]

    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Processed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

print(f"\nDone! Collection now has {collection.count()} chunks stored.")