import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from query import _bm25_search, _embedding_search

question = "What was Microsofts net income for fiscal year 2026, and how does it compare to fiscal year 2025?"
target = "microsoft_554"

bm25_ids = _bm25_search(question, company_filter="microsoft", n_results=100)
embedding_ids = _embedding_search(question, company_filter="microsoft", n_results=100)

bm25_rank = bm25_ids.index(target) + 1 if target in bm25_ids else "not in top 100"
embedding_rank = embedding_ids.index(target) + 1 if target in embedding_ids else "not in top 100"

print(f"BM25 rank for {target}: {bm25_rank}")
print(f"Embedding rank for {target}: {embedding_rank}")

print("\nTop 10 BM25:", bm25_ids[:10])
print("Top 10 embedding:", embedding_ids[:10])