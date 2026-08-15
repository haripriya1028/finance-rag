from sentence_transformers import SentenceTransformer
import chromadb
import os
import json
import re
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found - check your .env file exists in the project root")

client_genai = genai.Client(api_key=api_key)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="finance_filings")

# --- BM25 setup (loaded once at startup) ---
with open("data/chunks.json", "r", encoding="utf-8") as f:
    _all_chunks = json.load(f)

_chunk_by_id = {c["chunk_id"]: c for c in _all_chunks}
_bm25_corpus_ids = [c["chunk_id"] for c in _all_chunks]


def _tokenize(text):
    """Strip table/formatting noise (|, $) before splitting into tokens,
    so keywords like 'net' 'income' '133,749' aren't diluted by symbols."""
    text = re.sub(r"[|$]", " ", text.lower())
    return text.split()


_bm25_tokenized = [_tokenize(c["text"]) for c in _all_chunks]
_bm25_index = BM25Okapi(_bm25_tokenized)

print(f"BM25 index built over {len(_all_chunks)} chunks")


def _bm25_search(question, company_filter=None, n_results=30):
    """Return a ranked list of chunk_ids using BM25 keyword search."""
    tokenized_query = _tokenize(question)
    scores = _bm25_index.get_scores(tokenized_query)

    scored = []
    for chunk_id, score in zip(_bm25_corpus_ids, scores):
        chunk = _chunk_by_id[chunk_id]
        if company_filter and chunk["company"] != company_filter:
            continue
        scored.append((chunk_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in scored[:n_results]]


def _embedding_search(question, company_filter=None, n_results=30):
    """Return a ranked list of chunk_ids using Chroma embedding search."""
    question_embedding = model.encode([question]).tolist()
    where_clause = {"company": company_filter} if company_filter else None
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
        where=where_clause
    )
    return results["ids"][0]


def _reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Combine multiple ranked lists of chunk_ids into one fused ranking.
    RRF score for an id = sum over lists of 1 / (k + rank_in_that_list).
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in fused]


def retrieve_chunks(question, top_k=5, company_filter=None):
    candidate_pool = max(top_k * 3, 30)

    bm25_ids = _bm25_search(question, company_filter, n_results=candidate_pool)
    embedding_ids = _embedding_search(question, company_filter, n_results=candidate_pool)

    fused_ids = _reciprocal_rank_fusion([bm25_ids, embedding_ids])[:top_k]

    documents = [_chunk_by_id[cid]["text"] for cid in fused_ids]
    metadatas = [{"company": _chunk_by_id[cid]["company"], "chunk_id": cid} for cid in fused_ids]

    return {"documents": [documents], "metadatas": [metadatas], "ids": [fused_ids]}


def retrieve_chunks_multi_company(question, top_k_per_company=5, companies=["microsoft", "nvidia", "jpmc"]):
    all_docs = []
    all_metadatas = []
    all_ids = []

    for company in companies:
        result = retrieve_chunks(question, top_k=top_k_per_company, company_filter=company)
        all_docs.extend(result["documents"][0])
        all_metadatas.extend(result["metadatas"][0])
        all_ids.extend(result["ids"][0])

    return {"documents": [all_docs], "metadatas": [all_metadatas], "ids": [all_ids]}


def generate_answer(question, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks["documents"][0])

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

    response = client_genai.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    question = input("Ask a question about the financial filings: ")
    company = input("Filter by company (microsoft/nvidia/jpmc, or leave blank for all): ").strip().lower()
    company = company if company else None

    print("\nRetrieving relevant chunks...")
    results = retrieve_chunks(question, top_k=15, company_filter=company)

    print("\nSources used:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"  [{i+1}] {meta['company']}: {doc[:100]}...")

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)
    print("ANSWER:")
    print(answer)