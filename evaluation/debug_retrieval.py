import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from query import retrieve_chunks_multi_company, generate_answer

question = "Which of the three companies (Microsoft, NVIDIA, JPMorgan Chase) had the highest total revenue in their most recent fiscal year?"

retrieved = retrieve_chunks_multi_company(question, top_k_per_company=5)

print("Chunks retrieved:")
for i, (doc, meta) in enumerate(zip(retrieved["documents"][0], retrieved["metadatas"][0])):
    print(f"  [{i+1}] {meta['company']}: {doc[:100]}...")

print(f"\nTotal chunks: {len(retrieved['documents'][0])}")

print("\nGenerating answer...")
answer = generate_answer(question, retrieved)
print("\nANSWER:", answer)