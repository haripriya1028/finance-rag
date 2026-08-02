from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found - check your .env file exists in the project root")

client_genai = genai.Client(api_key=api_key)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="finance_filings")

def retrieve_chunks(question, top_k=5, company_filter=None):
    question_embedding = model.encode([question]).tolist()
    where_clause = {"company": company_filter} if company_filter else None
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        where=where_clause
    )
    return results

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