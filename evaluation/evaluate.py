import json
import sys
import os
import time
from google.genai.errors import ServerError, ClientError

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from query import retrieve_chunks, retrieve_chunks_multi_company, generate_answer

with open("evaluation/test_questions.json", "r", encoding="utf-8") as f:
    test_questions = json.load(f)

# --- load cached answers from previous runs, if any ---
cache_path = "evaluation/answer_cache.json"
answer_cache = {}
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        answer_cache = json.load(f)
    print(f"Loaded {len(answer_cache)} cached answers from previous run(s)")

results = []

for test in test_questions:
    question = test["question"]
    if test["company"] == "all":
        retrieved = retrieve_chunks_multi_company(question, top_k_per_company=5)
    else:
        retrieved = retrieve_chunks(question, top_k=15, company_filter=test["company"])

    # --- use cached answer if we already have one for this question id ---
    if test["id"] in answer_cache and not answer_cache[test["id"]].startswith("ERROR"):
        answer = answer_cache[test["id"]]
        print(f"[cached] {test['id']}")
    else:
        answer = None
        for attempt in range(5):
            try:
                answer = generate_answer(question, retrieved)
                break
            except (ServerError, ClientError) as e:
                wait = 15
                print(f"API error, retrying in {wait}s... (attempt {attempt+1}/5): {e}")
                time.sleep(wait)
        if answer is None:
            answer = "ERROR: failed after retries"

        if not answer.startswith("ERROR"):
            answer_cache[test["id"]] = answer
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(answer_cache, f, indent=2)

        time.sleep(6)

    retrieved_companies = list(set(m["company"] for m in retrieved["metadatas"][0]))

    keywords = test["expected_keywords"]
    if keywords:
        matched_keywords = [kw for kw in keywords if kw.lower() in answer.lower()]
        keyword_score = len(matched_keywords) / len(keywords)
        passed = keyword_score >= 0.5
    else:
        matched_keywords = []
        keyword_score = None
        passed = None

    print(f"Answer: {answer[:200]}")
    print(f"Expected: {test['expected_answer']}")
    print(f"Retrieved from: {retrieved_companies}")
    print(f"Keywords matched: {matched_keywords}/{keywords}")
    print(f"PASSED: {passed}")

    results.append({
        "id": test["id"],
        "company": test["company"],
        "difficulty": test["difficulty"],
        "question": question,
        "generated_answer": answer,
        "expected_answer": test["expected_answer"],
        "retrieved_companies": retrieved_companies,
        "keyword_score": keyword_score,
        "passed": passed
    })

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

auto_scored = [r for r in results if r["passed"] is not None]
passed_count = sum(1 for r in auto_scored if r["passed"])
print(f"Auto-scored questions: {len(auto_scored)}")
print(f"Passed: {passed_count}/{len(auto_scored)} ({passed_count/len(auto_scored)*100:.1f}%)")

for diff in ["easy", "medium", "hard"]:
    diff_results = [r for r in auto_scored if r["difficulty"] == diff]
    if diff_results:
        diff_passed = sum(1 for r in diff_results if r["passed"])
        print(f"  {diff}: {diff_passed}/{len(diff_results)}")

with open("evaluation/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nFull results saved to evaluation/results.json")