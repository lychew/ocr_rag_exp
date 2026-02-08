#!/usr/bin/env python3
"""
Setup - Build vector database collections.

This script runs: PDF -> OCR -> Chunking -> Embedding -> ChromaDB

Usage:
    # Setup best config only (Tesseract + Semantic) - fastest
    uv run scripts/setup.py

    # Setup all 9 combinations (3 OCR x 3 chunking) - takes ~30 min
    uv run scripts/setup.py --all

    # Setup specific OCR (all 3 chunking strategies for that OCR)
    uv run scripts/setup.py --ocr tesseract
    uv run scripts/setup.py --ocr easyocr
    uv run scripts/setup.py --ocr paddleocr

    # Setup specific combination
    uv run scripts/setup.py --ocr tesseract --chunking semantic
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


OCR_MODELS = ["tesseract", "easyocr", "paddleocr"]
CHUNKING_STRATEGIES = ["page", "parent_child", "semantic"]

# Best performing configuration (tesseract with preprocessing)
DEFAULT_OCR = "tesseract"
DEFAULT_CHUNKING = "semantic"


def setup_collection(ocr: str, chunking: str) -> int:
    """Setup a single collection. Returns number of chunks indexed."""
    collection_name = f"{ocr}__{chunking}"
    print(f"\n{'='*60}")
    print(f"Setting up: {collection_name}")
    print("=" * 60)

    pipeline = RAGPipeline(ocr_name=ocr, chunking_name=chunking)

    # Check if already ingested
    count = pipeline.vectorstore.count()
    if count > 0:
        print(f"  Already has {count} chunks. Skipping.")
        return count

    print("  Running ingest pipeline...")
    n_chunks = pipeline.ingest()
    print(f"  Done! {n_chunks} chunks indexed.")
    return n_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Setup vector database collections for RAG pipeline"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Setup all 9 combinations (3 OCR x 3 chunking)"
    )
    parser.add_argument(
        "--ocr",
        choices=OCR_MODELS,
        help="OCR model to use (sets up all chunking strategies for this OCR)"
    )
    parser.add_argument(
        "--chunking",
        choices=CHUNKING_STRATEGIES,
        help="Chunking strategy (requires --ocr)"
    )

    args = parser.parse_args()

    # Determine what to setup
    if args.all:
        # Setup all 9 combinations
        combinations = [(ocr, chunk) for ocr in OCR_MODELS for chunk in CHUNKING_STRATEGIES]
        print("Setting up ALL 9 combinations...")
        print("This will take ~30 minutes.\n")

    elif args.ocr and args.chunking:
        # Setup specific combination
        combinations = [(args.ocr, args.chunking)]

    elif args.ocr:
        # Setup all chunking strategies for specific OCR
        combinations = [(args.ocr, chunk) for chunk in CHUNKING_STRATEGIES]
        print(f"Setting up all chunking strategies for {args.ocr}...")

    else:
        # Default: setup best config only
        combinations = [(DEFAULT_OCR, DEFAULT_CHUNKING)]
        print(f"Setting up best configuration: {DEFAULT_OCR} + {DEFAULT_CHUNKING}")
        print("(Use --all for all combinations, or --ocr/--chunking for specific ones)")

    # Run setup for each combination
    total_chunks = 0
    for ocr, chunking in combinations:
        try:
            n = setup_collection(ocr, chunking)
            total_chunks += n
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"Collections ready: {len(combinations)}")
    print(f"Total chunks indexed: {total_chunks}")
    print("\nYou can now run:")
    print("  uv run scripts/2_answer_questions.py 'What are germs?'")
    print("  uv run scripts/3_evaluate.py")


if __name__ == "__main__":
    main()
