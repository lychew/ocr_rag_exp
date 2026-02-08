#!/usr/bin/env python3
"""
Evaluate - Run evaluation metrics on RAG pipeline answers.

Evaluates answers with quality scores (groundedness, faithfulness, relevance).

Usage:
    # Evaluate a custom question
    uv run scripts/evaluate.py --question "What are germs?"

    # Evaluate all 3 target questions (default)
    uv run scripts/evaluate.py

    # Evaluate with specific OCR + chunking
    uv run scripts/evaluate.py --ocr tesseract --chunking semantic

    # Use different prompt strategies
    uv run scripts/evaluate.py --prompt cot
    uv run scripts/evaluate.py --prompt few_shot

    # Choose evaluation method
    uv run scripts/evaluate.py --eval-method groundedness
    uv run scripts/evaluate.py --eval-method ragas
    uv run scripts/evaluate.py --eval-method all

    # Save results to file
    uv run scripts/evaluate.py --output results/evaluation.json

Prompt strategies:
    - basic:    Simple prompt with grounding instructions (default)
    - cot:      Chain-of-Thought - step-by-step reasoning
    - few_shot: Few-shot - includes example Q&A pair

Evaluation methods:
    - simple:       Confidence score only (free, fast)
    - groundedness: LLM-as-judge checks claims (recommended)
    - ragas:        Full Ragas metrics (faithfulness, relevance)
    - all:          Both groundedness and ragas
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.evaluation.evaluator import Evaluator, EvaluatorConfig


OCR_MODELS = ["tesseract", "easyocr", "paddleocr"]
CHUNKING_STRATEGIES = ["page", "parent_child", "semantic"]
PROMPT_STRATEGIES = ["basic", "cot", "few_shot"]
EVAL_METHODS = ["simple", "groundedness", "ragas", "all"]

# Best performing configuration (tesseract with preprocessing)
DEFAULT_OCR = "tesseract"
DEFAULT_CHUNKING = "semantic"
DEFAULT_PROMPT = "basic"
DEFAULT_EVAL_METHOD = "groundedness"


def evaluate_single_question(pipeline, question: str, eval_method: str):
    """Evaluate a single question and return the result."""
    evaluator = Evaluator(EvaluatorConfig(method=eval_method))

    response, retrieved = pipeline.answer_question(question)
    evaluated = evaluator.evaluate(response, retrieved)

    return evaluated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline with quality metrics"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Custom question to evaluate (default: use 3 target questions)"
    )
    parser.add_argument(
        "--ocr",
        choices=OCR_MODELS,
        default=DEFAULT_OCR,
        help=f"OCR model to use (default: {DEFAULT_OCR})"
    )
    parser.add_argument(
        "--chunking",
        choices=CHUNKING_STRATEGIES,
        default=DEFAULT_CHUNKING,
        help=f"Chunking strategy to use (default: {DEFAULT_CHUNKING})"
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPT_STRATEGIES,
        default=DEFAULT_PROMPT,
        help=f"Prompt strategy: basic, cot (chain-of-thought), few_shot (default: {DEFAULT_PROMPT})"
    )
    parser.add_argument(
        "--eval-method",
        choices=EVAL_METHODS,
        default=DEFAULT_EVAL_METHOD,
        help=f"Evaluation method (default: {DEFAULT_EVAL_METHOD})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Evaluating RAG Pipeline")
    print(f"  OCR:        {args.ocr}")
    print(f"  Chunking:   {args.chunking}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  Eval:       {args.eval_method}")
    print("=" * 60)

    pipeline = RAGPipeline(
        ocr_name=args.ocr,
        chunking_name=args.chunking,
        prompt_strategy=args.prompt
    )

    if pipeline.vectorstore.count() == 0:
        print(f"\nERROR: Collection '{args.ocr}__{args.chunking}' is empty.")
        print(f"Run setup first:")
        print(f"  uv run scripts/setup.py --ocr {args.ocr} --chunking {args.chunking}")
        sys.exit(1)

    print(f"\nCollection has {pipeline.vectorstore.count()} chunks.")

    # Evaluate custom question or all target questions
    if args.question:
        print(f"Evaluating custom question...\n")
        results = [evaluate_single_question(pipeline, args.question, args.eval_method)]
    else:
        print("Evaluating 3 target questions...\n")
        results = pipeline.answer_all_questions(
            with_evaluation=True,
            eval_method=args.eval_method
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    total_groundedness = 0
    total_faithfulness = 0
    total_relevance = 0
    total_confidence = 0

    for i, r in enumerate(results, 1):
        e = r.evaluation
        total_groundedness += e.groundedness
        total_faithfulness += e.faithfulness
        total_relevance += e.relevance
        total_confidence += (e.confidence or 0)

        q_text = r.response.question
        if len(q_text) > 60:
            q_text = q_text[:60] + "..."

        print(f"\nQ{i}: {q_text}")
        print(f"A:  {r.response.answer[:100]}...")
        print(f"    Groundedness: {e.groundedness:.2f}")
        print(f"    Faithfulness: {e.faithfulness:.2f}")
        print(f"    Relevance:    {e.relevance:.2f}")
        print(f"    Confidence:   {e.confidence:.2f}" if e.confidence else "    Confidence:   N/A")

    n = len(results)
    if n > 1:
        print("\n" + "-" * 60)
        print("AVERAGES:")
        print(f"    Groundedness: {total_groundedness/n:.2f}")
        print(f"    Faithfulness: {total_faithfulness/n:.2f}")
        print(f"    Relevance:    {total_relevance/n:.2f}")
        print(f"    Confidence:   {total_confidence/n:.2f}")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        output = [r.model_dump() for r in results]
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
