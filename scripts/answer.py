#!/usr/bin/env python3
"""
Answer - Answer any question about the book (no evaluation scores).

Usage:
    # Answer a custom question (uses best config by default)
    uv run scripts/answer.py --question "What are germs?"

    # Answer with specific OCR + chunking
    uv run scripts/answer.py --question "What are germs?" --ocr tesseract --chunking page

    # Answer with different prompt strategies
    uv run scripts/answer.py --question "What are germs?" --prompt cot
    uv run scripts/answer.py --question "What are germs?" --prompt few_shot

    # Answer all 3 target questions from the assignment
    uv run scripts/answer.py --all

    # Save output to file
    uv run scripts/answer.py --all --output results/answers.json

Prompt strategies:
    - basic:    Simple prompt with grounding instructions (default)
    - cot:      Chain-of-Thought - step-by-step reasoning
    - few_shot: Few-shot - includes example Q&A pair

Output format matches assignment requirements:
{
    "question": "...",
    "answer": "...",
    "supporting_chunks": [{"chunk_id": "...", "page": ...}]
}
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


OCR_MODELS = ["tesseract", "easyocr", "paddleocr"]
CHUNKING_STRATEGIES = ["page", "parent_child", "semantic"]
PROMPT_STRATEGIES = ["basic", "cot", "few_shot"]

# Best performing configuration (tesseract with preprocessing)
DEFAULT_OCR = "tesseract"
DEFAULT_CHUNKING = "semantic"
DEFAULT_PROMPT = "basic"

# Target questions from assignment
TARGET_QUESTIONS = [
    "What are the main ways to fight disease germs according to the book?",
    "How does the book describe the importance of pure air and its effect on health?",
    "Based on the principles described in the book, explain why preventing germs from entering the body and maintaining a clean environment together are more effective than either measure alone in reducing disease.",
]


def answer_question(question: str, ocr: str, chunking: str, prompt: str) -> dict:
    """
    Answer a question about the Principles of Public Health book.

    Args:
        question: The question to answer
        ocr: OCR model to use
        chunking: Chunking strategy to use
        prompt: Prompt strategy to use

    Returns:
        dict with format:
        {
            "question": "...",
            "answer": "...",
            "supporting_chunks": [{"chunk_id": "...", "page": ...}]
        }
    """
    pipeline = RAGPipeline(ocr_name=ocr, chunking_name=chunking, prompt_strategy=prompt)

    # Check if database exists
    if pipeline.vectorstore.count() == 0:
        print(f"ERROR: Collection '{ocr}__{chunking}' is empty.", file=sys.stderr)
        print(f"Run setup first:", file=sys.stderr)
        print(f"  uv run scripts/setup.py --ocr {ocr} --chunking {chunking}", file=sys.stderr)
        sys.exit(1)

    # Answer the question
    response, _ = pipeline.answer_question(question)

    return response.model_dump()


def main():
    parser = argparse.ArgumentParser(
        description="Answer questions about the Principles of Public Health book"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="The question to answer"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Answer all 3 target questions from the assignment"
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
        "--output", "-o",
        type=str,
        help="Save output to JSON file"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.question and not args.all:
        parser.error("Either --question or --all is required")

    # Answer questions
    if args.all:
        print(f"Answering all {len(TARGET_QUESTIONS)} target questions...", file=sys.stderr)
        print(f"Config: {args.ocr} + {args.chunking} + {args.prompt} prompt\n", file=sys.stderr)

        results = []
        for i, q in enumerate(TARGET_QUESTIONS, 1):
            print(f"[{i}/{len(TARGET_QUESTIONS)}] {q[:50]}...", file=sys.stderr)
            result = answer_question(q, args.ocr, args.chunking, args.prompt)
            results.append(result)

        output = results
    else:
        print(f"Config: {args.ocr} + {args.chunking} + {args.prompt} prompt", file=sys.stderr)
        output = answer_question(args.question, args.ocr, args.chunking, args.prompt)

    # Output result
    json_output = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"\nSaved to: {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
