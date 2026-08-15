import sys
import os
import time
from google.genai.errors import ServerError, ClientError

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from query import retrieve_chunks, generate_answer

test_cases = [
    {
        "question": "What percentage of NVIDIA's revenue came from its largest direct customer in fiscal year 2026?",
        "company": "nvidia",
        "target_chunk_ids": ["nvidia_674", "nvidia_1027"]
    },
    {
        "question": "What was Microsofts net income for fiscal year 2026, and how does it compare to fiscal year 2025?",
        "company": "microsoft",
        "target_chunk_ids": ["microsoft_554"]
    }
]

for case in test_cases:
    print("="*70)
    print("QUESTION:", case["question"])
    print("="*70)

    retrieved = retrieve_chunks(case["question"], top_k=15, company_filter=case["company"])

    print("\nTop 15 retrieved chunk_ids:")
    print(retrieved["ids"][0])

    found = [cid for cid in case["target_chunk_ids"] if cid in retrieved["ids"][0]]
    print(f"\nTarget chunk(s) {case['target_chunk_ids']} found in top 15: {found if found else 'NONE'}")

    print("\nGenerating answer...")
    answer = None
    for attempt in range(3):
        try:
            answer = generate_answer(case["question"], retrieved)
            break
        except (ServerError, ClientError) as e:
            print(f"API error ({e}), retrying in 10s... (attempt {attempt+1}/3)")
            time.sleep(10)
    if answer is None:
        answer = "ERROR: failed after retries"

    print("ANSWER:", answer)
    print("\n")