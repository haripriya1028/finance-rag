from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

filings_dir = "data/filings"
all_chunks = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

for filename in os.listdir(filings_dir):
    if filename.endswith("_extracted.txt"):
        filepath = os.path.join(filings_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        company_name = filename.replace("_extracted.txt", "")
        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "company": company_name,
                "chunk_id": f"{company_name}_{i}",
                "text": chunk
            })

        print(f"{company_name}: {len(chunks)} chunks created")

print(f"\nTotal chunks across all filings: {len(all_chunks)}")

os.makedirs("data", exist_ok=True)
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print("Saved to data/chunks.json")