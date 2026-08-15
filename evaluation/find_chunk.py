import json

with open("data/chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

for chunk in all_chunks:
    if chunk["company"] == "microsoft" and "133,749" in chunk["text"]:
        print(chunk["chunk_id"])
        print(chunk["text"])
        print("---")