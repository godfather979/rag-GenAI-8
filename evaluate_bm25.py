"""
BM25-RAG evaluation runner.

Requires bm25_rag.py in the same directory. That module should expose:
    nodes, tokenizer, model, generate_answer

Outputs:
    evaluation_results.json  - complete results including retrieved chunk text
    evaluation_results.csv   - spreadsheet-friendly summary

Usage:
    python evaluate_bm25.py
    python evaluate_bm25.py --k 3
    python evaluate_bm25.py --all-k
"""

import argparse
import csv
import json
import time
from pathlib import Path

from llama_index.retrievers.bm25 import BM25Retriever

# Importing bm25_rag.py loads the existing index and local Qwen model.
from bm25_rag import nodes, generate_answer


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "evaluation_questions.json"
JSON_OUTPUT = BASE_DIR / "evaluation_results.json"
CSV_OUTPUT = BASE_DIR / "evaluation_results.csv"


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["questions"]


def source_info(node):
    md = node.metadata or {}
    return {
        "source": md.get("source", ""),
        "title": md.get("source_title", ""),
        "url": md.get("source_url", ""),
        "type": md.get("source_type", ""),
        "page": md.get("page", ""),
    }


def run_evaluation(k_values):
    questions = load_questions()
    all_results = []

    for k in k_values:
        print(f"\n{'=' * 72}\nRunning evaluation with top_k={k}\n{'=' * 72}")

        retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=k,
        )

        for item in questions:
            qid = item["id"]
            question = item["question"]
            started = time.perf_counter()

            try:
                retrieved = retriever.retrieve(question)
                retrieval_done = time.perf_counter()

                answer, sources = generate_answer(question, retrieved)
                finished = time.perf_counter()

                retrieved_chunks = []
                for rank, result in enumerate(retrieved, 1):
                    retrieved_chunks.append({
                        "rank": rank,
                        "bm25_score": result.score,
                        "source_metadata": source_info(result.node),
                        "text": result.node.get_content(),
                    })

                # Hit@k is only automatically calculable when source keywords
                # have been manually added to the JSON question.
                keywords = [
                    x.lower() for x in item.get("relevant_source_keywords", [])
                ]
                hit_at_k = None
                if keywords:
                    hit_at_k = any(
                        any(
                            kw in str(chunk["source_metadata"]).lower()
                            for kw in keywords
                        )
                        for chunk in retrieved_chunks
                    )

                result_record = {
                    "question_id": qid,
                    "category": item["category"],
                    "question": question,
                    "top_k": k,
                    "expected_answer": item.get("expected_answer", ""),
                    "answerable_expected": item.get("answerable"),
                    "generated_answer": answer,
                    "retrieval_hit_at_k": hit_at_k,
                    "retrieval_time_seconds": round(
                        retrieval_done - started, 4
                    ),
                    "generation_time_seconds": round(
                        finished - retrieval_done, 4
                    ),
                    "total_time_seconds": round(finished - started, 4),
                    "retrieved_chunks": retrieved_chunks,
                    "sources_returned_by_generator": sources,
                    "manual_answer_correct": "",
                    "manual_faithfulness": "",
                    "manual_abstention_correct": "",
                    "notes": "",
                }

                all_results.append(result_record)
                print(f"\n[{qid} | k={k}] {question}")
                print(f"Answer: {answer[:500]}")
                print(
                    f"Retrieved={len(retrieved)} | "
                    f"Time={result_record['total_time_seconds']}s"
                )

                # Save incrementally, so progress is retained if interrupted.
                JSON_OUTPUT.write_text(
                    json.dumps(all_results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            except Exception as exc:
                error_record = {
                    "question_id": qid,
                    "category": item["category"],
                    "question": question,
                    "top_k": k,
                    "error": repr(exc),
                    "manual_answer_correct": "",
                    "manual_faithfulness": "",
                    "manual_abstention_correct": "",
                    "notes": "",
                }
                all_results.append(error_record)
                JSON_OUTPUT.write_text(
                    json.dumps(all_results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"ERROR on {qid}: {exc}")

    # CSV summary: keep full retrieved passages in JSON to avoid a huge CSV.
    columns = [
        "question_id", "category", "question", "top_k",
        "expected_answer", "answerable_expected", "generated_answer",
        "retrieval_hit_at_k", "retrieval_time_seconds",
        "generation_time_seconds", "total_time_seconds",
        "manual_answer_correct", "manual_faithfulness",
        "manual_abstention_correct", "notes", "error"
    ]

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print("\nEvaluation finished.")
    print(f"JSON results: {JSON_OUTPUT}")
    print(f"CSV results:  {CSV_OUTPUT}")
    print(f"Rows recorded: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k", type=int, choices=[1, 3, 5, 10],
        help="Run one top-k setting only."
    )
    parser.add_argument(
        "--all-k", action="store_true",
        help="Run all required settings: k=1,3,5,10 (120 generations)."
    )
    args = parser.parse_args()

    if args.all_k:
        k_values = [1, 3, 5, 10]
    else:
        # A single k=3 run is the default; use --all-k for the full comparison.
        k_values = [args.k or 3]

    run_evaluation(k_values)


if __name__ == "__main__":
    main()
