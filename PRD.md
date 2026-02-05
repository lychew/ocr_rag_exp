# PRD: RAG Pipeline for Scanned PDF Book

## Project Overview

Build a RAG (Retrieval-Augmented Generation) pipeline for answering questions about a scanned PDF book ("Principles of Public Health" from Project Gutenberg). The PDF is image-based (123 pages, no selectable text) and requires OCR before text processing.

## Source Data

- `GenAI_Assignment_Health_v1/Principles of Public Health.pdf` — 123-page scanned PDF
- `GenAI_Assignment_Health_v1/GenAI Assignment.docx` — Assignment specification

## Target Questions

1. What are the main ways to fight disease germs according to the book?
2. How does the book describe the importance of pure air and its effect on health?
3. Based on the principles described in the book, explain why preventing germs from entering the body and maintaining a clean environment together are more effective than either measure alone in reducing disease.

---

## System Architecture

```
User Query → Retriever → Prompt Builder → LLM → Pydantic Output
                ↑                                      ↓
PDF → OCR → Chunker → Embedder → VectorStore    Evaluator → Langfuse
```

### Per-Page Processing (Ingest)

```
Page image
  ├── Layout Detection (PaddleOCR or built-in)
  │     ├── Text regions → OCR engine → clean text → text chunks
  │     └── Figure regions → crop → VLM description → figure chunks
  │
  └── Both chunk types → embed → ChromaDB collection "{model}_{strategy}"
```

---

## Experiment Design

### Two Entry Points

1. **Ingest script** — For each (OCR model × chunking strategy) combination: run OCR, chunk, embed, store in a dedicated ChromaDB collection.
2. **Evaluate script** — For each collection: run the 3 target questions, retrieve chunks, generate answers, score with Ragas, log to Langfuse.

### Model Capabilities

| Model | Extracts Text | Detects Layout (figures vs text) | Describes Diagrams |
|-------|:---:|:---:|:---:|
| Tesseract | Yes | No | No |
| EasyOCR | Yes | No | No |
| PaddleOCR | Yes | Yes (built-in) | No |
| Surya | Yes | Yes (built-in) | No |
| Qwen2.5-VL | Yes | Yes | Yes |
| GPT-4o (API) | Yes | Yes | Yes |

### How Each Model Handles Diagrams

**Tesseract / EasyOCR** (no layout awareness):
```
Page image → PaddleOCR layout detector →
    Text regions → Tesseract/EasyOCR → text chunks
    Figure regions → GPT-4o or Qwen2.5-VL → figure description chunks
```

**PaddleOCR / Surya** (layout aware, can't describe):
```
Page image → PaddleOCR/Surya (text + layout) →
    Text regions → text chunks
    Figure regions → GPT-4o or Qwen2.5-VL → figure description chunks
```

**Qwen2.5-VL / GPT-4o** (does everything):
```
Page image → VLM → structured text + figure descriptions → all chunks
```

### Experiment Matrix

#### OCR Track (4 models + shared figure description via GPT-4o)

| # | OCR Model | Layout Detection | Figure Description |
|---|-----------|-----------------|-------------------|
| 1 | Tesseract | PaddleOCR layout | GPT-4o |
| 2 | EasyOCR | PaddleOCR layout | GPT-4o |
| 3 | PaddleOCR | Built-in | GPT-4o |
| 4 | Surya | Built-in | GPT-4o |

#### VLM Track (1 model, does everything)

| # | Model | Layout | Figure Description |
|---|-------|--------|-------------------|
| 5 | Qwen2.5-VL-7B | Built-in | Built-in |

#### Chunking Strategies (crossed with each model above)

| # | Strategy | Description |
|---|----------|-------------|
| A | Page-level | 1 chunk per page (~123 chunks), baseline |
| B | Parent-child | Parent chunks (800-1500 tokens) contain child chunks (200-400 tokens with overlap). Search children for precision, return parents for LLM context |
| C | Semantic | Uses embedding similarity between sentences to find natural breakpoints |

#### Fixed Across All Experiments

- Embedding: **BAAI/bge-small-en-v1.5**
- Vector store: **ChromaDB** (one collection per combo, e.g. `tesseract__parent_child`)
- LLM for answers: **GPT-4o-mini**
- Retrieval: **top-k = 5, cosine similarity**

#### Total: 5 models × 3 chunking strategies = 15 combinations

---

## OCR Post-Processing

After OCR produces raw text, a post-processing step cleans it before chunking:

| Issue | Example | Fix |
|-------|---------|-----|
| Spurious line breaks | `"The importance of\nclean water is"` | Join lines not ending with sentence punctuation |
| Headers/footers | `"PRINCIPLES OF PUBLIC HEALTH    47"` repeated | Detect repeated strings across pages, strip them |
| Hyphenated words | `"preven-\ntion"` → `"prevention"` | Rejoin words split across lines |
| OCR noise | `"th e"`, `"dis ease"`, stray `"\|"` | Regex patterns, merge split words |
| Multiple spaces | `"The    grounds   of"` | Collapse to single space |
| Page numbers | `"— 42 —"` or standalone `"47"` | Strip isolated numbers at top/bottom |

---

## Data Caching Strategy

Each stage is cached independently so re-runs are fast:

1. **Page images** — extracted once from PDF, reused by all OCR models → `data/pages/`
2. **OCR output** — saved per model → `data/ocr_output/{model_name}/`
3. **Chunks** — saved per (OCR + chunking) combo → `data/chunks/{model}_{strategy}/`
4. **Vector store collections** — one per combo inside single ChromaDB → `data/vectordb/`

---

## Output Schema

Every answer must conform to this structure:

```python
class SupportingChunk(BaseModel):
    chunk_id: str
    page: int

class RAGResponse(BaseModel):
    question: str
    answer: str
    supporting_chunks: list[SupportingChunk]

class EvaluationScore(BaseModel):
    groundedness: float   # 0.0-1.0
    relevance: float      # 0.0-1.0
    faithfulness: float   # 0.0-1.0
    confidence: float | None = None

class EvaluatedResponse(BaseModel):
    response: RAGResponse
    evaluation: EvaluationScore
```

---

## Evaluation Metrics

### RAG Quality (primary evaluation)

| Metric | What it measures | How |
|--------|-----------------|-----|
| Faithfulness | Every claim in answer supported by retrieved chunks? | LLM-as-judge checks each sentence against source chunks |
| Answer Relevance | Does the answer address the question? | LLM scores answer-question alignment |
| Context Precision | Are retrieved chunks actually relevant? | Were top-ranked chunks the ones containing the answer? |
| Groundedness | Can every statement be traced to a chunk? | LLM checks each claim has a source |
| Confidence Score | How similar were retrieved chunks to the query? | Average cosine similarity from vector search |

### Tooling

- **Ragas** — computes faithfulness, relevance, context precision
- **Langfuse** — traces full pipeline (latency, tokens, spans), stores Ragas scores for dashboard visualization

---

## CLI Interface

```bash
# Ingest all combinations
python -m src.main ingest --all

# Ingest one specific combination
python -m src.main ingest --ocr tesseract --chunking parent_child

# Evaluate all combinations against the 3 questions
python -m src.main evaluate --all

# Evaluate one combination
python -m src.main evaluate --ocr tesseract --chunking parent_child
```

---

## Project Structure

```
E:\GenAi_Assignment\
├── CLAUDE.md
├── PRD.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── config/
│   ├── default.yaml
│   ├── ocr_tesseract.yaml
│   ├── ocr_easyocr.yaml
│   ├── ocr_paddleocr.yaml
│   ├── ocr_surya.yaml
│   ├── vlm_qwen2vl.yaml
│   ├── chunking_page.yaml
│   ├── chunking_parent_child.yaml
│   └── chunking_semantic.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── factory.py
│   ├── pipeline.py
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── tesseract_ocr.py
│   │   ├── easyocr_extractor.py
│   │   ├── paddleocr_extractor.py
│   │   ├── surya_ocr.py
│   │   ├── vlm_extractor.py
│   │   └── preprocessing.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── page_chunker.py
│   │   ├── parent_child_chunker.py
│   │   └── semantic_chunker.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── sentence_transformer.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chroma_store.py
│   │   └── faiss_store.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dense_retriever.py
│   │   └── parent_child_retriever.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai_generator.py
│   │   └── prompt_templates.py
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py
│       ├── groundedness.py
│       └── langfuse_tracker.py
├── data/
│   ├── raw/
│   ├── pages/
│   ├── ocr_output/
│   ├── chunks/
│   └── vectordb/
├── notebooks/
│   ├── 01_ocr_comparison.ipynb
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_rag_pipeline_demo.ipynb
│   └── 04_evaluation_analysis.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_ocr.py
│   ├── test_chunking.py
│   ├── test_embedding.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_pipeline_integration.py
└── GenAI_Assignment_Health_v1/
    ├── Principles of Public Health.pdf
    └── GenAI Assignment.docx
```

---

## Implementation Phases

### Phase 1: Foundation
- Project scaffolding (pyproject.toml, .gitignore, .env.example)
- All Pydantic models in `src/models.py`
- Config loader + default YAML
- Factory/registry pattern
- Abstract base classes for all components

### Phase 2: OCR Pipeline
- PDF preprocessing (pdf2image, deskew, denoise)
- Layout detection (PaddleOCR layout module)
- Tesseract implementation
- PaddleOCR implementation
- Figure description via GPT-4o/Qwen2.5-VL
- OCR post-processing (line breaks, headers/footers, noise)
- Run on full 123-page PDF, save results

### Phase 3: Chunking
- Page chunker
- Parent-child chunker (with parent_id links)
- Unit tests for chunk correctness

### Phase 4: Embedding + Vector Store
- Sentence transformer embedding provider
- ChromaDB store with metadata
- Ingest pipeline: chunks → embeddings → store

### Phase 5: Retrieval + Generation
- Dense retriever
- Parent-child retriever (search children, return parents)
- Prompt templates
- OpenAI generator (gpt-4o-mini default)

### Phase 6: Integration
- Wire RAGPipeline orchestrator
- CLI entrypoint (main.py)
- Run 3 target questions, verify output format

### Phase 7: Evaluation
- Groundedness checker (LLM-as-judge)
- Langfuse integration (traces + scores)
- Ragas metrics (faithfulness, relevance, context precision)

### Phase 8: Additional OCR Engines + Notebooks
- EasyOCR, Surya OCR, Qwen2.5-VL implementations
- Comparison notebooks
- Semantic chunking

### Phase 9: Polish
- Error handling, edge cases
- Git init + initial commit

---

## Verification Plan

1. `python -m src.main ingest --all` — runs OCR → chunk → embed → store for all combos
2. `python -m src.main evaluate --all` — answers 3 questions per combo, outputs JSON
3. `pytest tests/` — all unit + integration tests pass
4. Check Langfuse dashboard for traces and evaluation scores
5. Verify JSON output matches the required schema
