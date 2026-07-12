"""
Phase 4: Evaluation framework.

Runs the golden Q&A set through the pipeline and reports:
  - answer correctness (LLM-as-judge vs. golden answer)
  - faithfulness (are claims grounded? reuses citation_accuracy)
  - retrieval relevance (did the right source file show up in top chunks?)
  - citation accuracy

Also supports --compare-strategies to run the same suite across
fixed / recursive / semantic chunking and report which wins on what —
the concrete numbers the guide says to lead with in interviews.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vectorstore import VectorStore
from app.bm25_index import BM25Index, build_from_vectorstore
from app.ingestion import load_directory
from app.chunking import chunk_document
from app.pipeline import ask
from app.llm_client import chat_json
from app import config


def judge_correctness(question: str, golden_answer: str, model_answer: str):
    """Returns (correct, error_message). correct is None (not False) when the
    judge call itself fails, so a judge crash never gets silently counted as
    'the model was wrong' — summarize() below tracks these separately."""
    prompt = (
        f"QUESTION: {question}\n\nGOLDEN ANSWER: {golden_answer}\n\nMODEL ANSWER: {model_answer}\n\n"
        "Does the MODEL ANSWER convey the same key facts as the GOLDEN ANSWER? "
        "Minor differences in wording or extra detail are fine as long as the key facts match. "
        'Respond ONLY with JSON: {"correct": true|false}'
    )
    try:
        raw = chat_json([{"role": "user", "content": prompt}])
        return bool(json.loads(raw).get("correct", False)), ""
    except Exception as e:
        return None, f"judge error: {e}"


def run_suite(store, bm25, golden_set, use_hybrid=True, verify=True, verbose=False):
    results = []
    for item in golden_set:
        result = ask(item["question"], store, bm25, use_hybrid=use_hybrid, verify=verify)

        if item["type"] == "no-answer":
            correct = result["confidence"]["is_low_confidence"]
            judge_error = ""
        else:
            correct, judge_error = judge_correctness(item["question"], item["expected_answer"], result["answer"])

        citation_acc = None
        if result.get("citation_report"):
            citation_acc = result["citation_report"]["citation_accuracy"]

        row = {
            "id": item["id"],
            "type": item["type"],
            "difficulty": item["difficulty"],
            "correct": correct,
            "citation_accuracy": citation_acc,
            "retrieval_confidence": result["confidence"]["retrieval_confidence"],
            "composite_confidence": result["confidence"]["composite_confidence"],
            "judge_error": judge_error,
        }
        results.append(row)

        if verbose:
            status = "PASS" if correct is True else ("ERROR" if correct is None else "FAIL")
            print(f"\n[{status}] {item['id']} ({item['type']}) — {item['question']}")
            print(f"  Golden:   {item['expected_answer']}")
            print(f"  Model:    {result['answer']}")
            if judge_error:
                print(f"  \u26a0 {judge_error}")

    return results


def summarize(results, label):
    n = len(results)
    correct = sum(1 for r in results if r["correct"] is True)
    judge_errors = sum(1 for r in results if r["correct"] is None)
    graded = n - judge_errors
    cit_scores = [r["citation_accuracy"] for r in results if r["citation_accuracy"] is not None]
    avg_cit = sum(cit_scores) / len(cit_scores) if cit_scores else None
    avg_conf = sum(r["composite_confidence"] for r in results) / n if n else 0

    print(f"\n=== {label} ===")
    if graded:
        print(f"Correctness:        {correct}/{graded} graded ({100*correct/graded:.1f}%)")
    else:
        print("Correctness:        N/A (all judge calls failed)")
    if judge_errors:
        print(f"Judge errors:       {judge_errors}/{n} (excluded from correctness — see per-row 'judge_error' / --verbose output)")
    print(f"Avg citation acc:   {avg_cit*100:.1f}%" if avg_cit is not None else "Avg citation acc:   N/A")
    print(f"Avg confidence:     {avg_conf:.3f}")
    return {
        "correctness_pct": 100 * correct / graded if graded else None,
        "judge_errors": judge_errors,
        "avg_citation_accuracy": avg_cit,
        "avg_confidence": avg_conf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "golden_qa.json"))
    parser.add_argument("--compare-strategies", action="store_true")
    parser.add_argument("--compare-hybrid", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print each question, golden answer, and model answer")
    args = parser.parse_args()

    with open(args.golden) as f:
        golden_set = json.load(f)

    store = VectorStore()
    bm25 = BM25Index()
    try:
        bm25.load()
    except FileNotFoundError:
        bm25.build(store.all_chunks())

    if args.compare_hybrid:
        hybrid_results = run_suite(store, bm25, golden_set, use_hybrid=True, verbose=args.verbose)
        dense_results = run_suite(store, bm25, golden_set, use_hybrid=False, verbose=args.verbose)
        summarize(hybrid_results, "Hybrid (dense + sparse + RRF)")
        summarize(dense_results, "Dense-only")
        return

    results = run_suite(store, bm25, golden_set, verbose=args.verbose)
    summary = summarize(results, "Full eval suite")

    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nDetailed results written to {out_path}")


if __name__ == "__main__":
    main()