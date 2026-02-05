# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG pipeline for answering questions about a scanned 123-page PDF book ("Principles of Public Health" from Project Gutenberg). The PDF is image-based (no selectable text) and requires OCR. The project runs 15 experiment combinations (5 OCR models x 3 chunking strategies), evaluates each with Ragas metrics, and logs traces to Langfuse.

Full specification: `PRD.md`

## Source Data

- `GenAI_Assignment_Health_v1/Principles of Public Health.pdf` — 123-page scanned PDF
- `GenAI_Assignment_Health_v1/GenAI Assignment.docx` — Assignment specification

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Ingest all 15 OCR/chunking combinations
python -m src.main ingest --all

# Ingest a single combination
python -m src.main ingest --ocr tesseract --chunking parent_child

# Evaluate all combinations against the 3 target questions
python -m src.main evaluate --all

# Evaluate a single combination
python -m src.main evaluate --ocr tesseract --chunking parent_child

# Run tests
pytest tests/
pytest tests/test_ocr.py              # single test file
pytest tests/test_ocr.py::test_name   # single test
```

## Architecture

### Pipeline Flow

```
PDF → Page Images → OCR/VLM → Post-process → Chunk → Embed → ChromaDB
                                                                  ↓
User Query → Retriever → Prompt Builder → GPT-4o-mini → JSON Answer → Ragas Eval → Langfuse
```

### Component Design

All components follow an **abstract base class + factory/registry pattern**:
- `src/ocr/base.py`, `src/chunking/base.py`, `src/embedding/base.py`, etc. define interfaces
- `src/factory.py` instantiates components by name from YAML config
- `src/config.py` loads and merges YAML configs from `config/`
- `src/models.py` contains all Pydantic models (`RAGResponse`, `EvaluatedResponse`, etc.)
- `src/pipeline.py` orchestrates the full ingest and evaluate flows

### OCR Track (5 models)

| Model | Layout Detection | Figure Description |
|-------|-----------------|-------------------|
| Tesseract | PaddleOCR layout | GPT-4o |
| EasyOCR | PaddleOCR layout | GPT-4o |
| PaddleOCR | Built-in | GPT-4o |
| Surya | Built-in | GPT-4o |
| Qwen2.5-VL-7B | Built-in | Built-in |

Models without layout awareness use PaddleOCR for layout detection. Figure regions are sent to a VLM (GPT-4o or Qwen2.5-VL) for description.

### Chunking Strategies (3)

| Strategy | Description |
|----------|-------------|
| Page-level | 1 chunk per page (~123 chunks) |
| Parent-child | Parent (800-1500 tokens) → child (200-400 tokens, overlap). Search children, return parents. |
| Semantic | Embedding similarity between sentences finds natural breakpoints |

### Fixed Across All Experiments

- **Embedding**: BAAI/bge-small-en-v1.5
- **Vector store**: ChromaDB — one collection per combo (e.g., `tesseract__parent_child`)
- **LLM**: GPT-4o-mini
- **Retrieval**: top-k=5, cosine similarity

### Data Caching

Each stage caches independently to avoid redundant work:
- `data/pages/` — page images from PDF
- `data/ocr_output/{model}/` — raw OCR results per model
- `data/chunks/{model}_{strategy}/` — chunked text per combination
- `data/vectordb/` — ChromaDB collections

## Output Schema

Every answer must conform to this structure:
```json
{
  "question": "...",
  "answer": "...",
  "supporting_chunks": [
    {"chunk_id": "...", "page": "..."}
  ]
}
```

## Target Questions

1. What are the main ways to fight disease germs according to the book?
2. How does the book describe the importance of pure air and its effect on health?
3. Based on the principles described in the book, explain why preventing germs from entering the body and maintaining a clean environment together are more effective than either measure alone in reducing disease.

## OCR Post-Processing

Raw OCR output requires cleaning before chunking: rejoin hyphenated words across lines, strip repeated headers/footers, collapse multiple spaces, remove stray page numbers, and join spurious line breaks mid-sentence. This logic lives in `src/ocr/preprocessing.py`.

## Environment

Requires a `.env` file with API keys (see `.env.example`). Key variables: `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
