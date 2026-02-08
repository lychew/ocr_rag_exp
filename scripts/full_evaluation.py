#!/usr/bin/env python3
"""
Full Evaluation - Compare all OCR + chunking combinations.

This runs the full experiment matrix:
- 3 OCR models: tesseract, easyocr, paddleocr
- 3 chunking strategies: page, parent_child, semantic
- Total: 9 combinations

Uses the 3 target questions as benchmark to compare all configurations.

Metrics evaluated:
- Groundedness: LLM-as-judge checks if answer is grounded in context
- Faithfulness: Ragas - are answer claims supported by context?
- Relevance: Ragas - is the answer relevant to the question?
- Confidence: Average retrieval similarity score

Takes ~45-90 minutes to run everything.

Usage:
    uv run scripts/full_evaluation.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


OCR_MODELS = ["tesseract", "easyocr", "paddleocr"]
CHUNKING_STRATEGIES = ["page", "parent_child", "semantic"]


def main():
    print("=" * 70)
    print("Running Full Experiment Matrix")
    print("3 OCR models x 3 chunking strategies = 9 combinations")
    print("=" * 70)

    results = {}

    for ocr in OCR_MODELS:
        for chunking in CHUNKING_STRATEGIES:
            combo = f"{ocr}__{chunking}"
            print(f"\n{'='*70}")
            print(f"Testing: {combo}")
            print("=" * 70)

            try:
                pipeline = RAGPipeline(ocr_name=ocr, chunking_name=chunking)

                # Check if needs ingest
                if pipeline.vectorstore.count() == 0:
                    print("  Ingesting...")
                    pipeline.ingest()

                # Evaluate with all metrics (groundedness + ragas)
                print("  Evaluating (groundedness + ragas)...")
                evals = pipeline.answer_all_questions(
                    with_evaluation=True,
                    eval_method="all"
                )

                # Calculate averages
                avg_g = sum(e.evaluation.groundedness for e in evals) / len(evals)
                avg_f = sum(e.evaluation.faithfulness for e in evals) / len(evals)
                avg_r = sum(e.evaluation.relevance for e in evals) / len(evals)
                avg_c = sum(e.evaluation.confidence or 0 for e in evals) / len(evals)

                results[combo] = {
                    "groundedness": round(avg_g, 2),
                    "faithfulness": round(avg_f, 2),
                    "relevance": round(avg_r, 2),
                    "confidence": round(avg_c, 2),
                }

                print(f"  Result: Groundedness={avg_g:.2f}, Faithfulness={avg_f:.2f}, Relevance={avg_r:.2f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results[combo] = {"error": str(e)}

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Ground':>8} {'Faith':>8} {'Relev':>8} {'Conf':>8}")
    print("-" * 70)

    # Sort by groundedness
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1]["groundedness"],
        reverse=True
    )

    for combo, scores in sorted_results:
        print(f"{combo:<25} {scores['groundedness']:>8.2f} {scores['faithfulness']:>8.2f} {scores['relevance']:>8.2f} {scores['confidence']:>8.2f}")

    # Show winner
    if sorted_results:
        winner = sorted_results[0]
        print("-" * 70)
        print(f"WINNER: {winner[0]} with {winner[1]['groundedness']:.2f} groundedness")

    # Save results
    output_path = Path(__file__).parent.parent / "results" / "experiment_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
