# RAG Pipeline for Principles of Public Health

A Retrieval-Augmented Generation (RAG) pipeline for answering questions about a scanned PDF book ("Principles of Public Health" from Project Gutenberg).

---

## Getting Started

### Step 1: Install Prerequisites

**Install uv (Python package manager):**
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install Tesseract OCR:**
```bash
# Windows - Download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki

# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr
```

### Step 2: Clone and Setup Environment

```bash
# Clone the repository
git clone <repo-url>
cd GenAi_Assignment

# Create virtual environment and install all dependencies
uv sync
```

### Step 3: Configure API Keys

**Option A: Create a `.env` file (Recommended)**
```bash
# Create .env file in project root
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Or copy the example and edit:
```bash
cp .env.example .env
# Then edit .env and add your OpenAI API key
```

**Option B: Set environment variable directly**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Windows CMD
set OPENAI_API_KEY=sk-your-key-here

# macOS/Linux
export OPENAI_API_KEY="sk-your-key-here"
```

> **Note:** Langfuse keys are optional (for observability tracking only).

### Step 4: Build the Vector Database

```bash
# Build with best configuration (Tesseract + Semantic) - ~5 minutes
uv run scripts/setup.py

# Or build all 9 OCR×Chunking combinations - ~30 minutes
uv run scripts/setup.py --all
```

### Step 5: Ask Questions

```bash
# Answer a question
uv run scripts/answer.py --question "What are the main ways to fight disease germs?"

# Answer all 3 target questions
uv run scripts/answer.py --all

# Answer with evaluation scores
uv run scripts/evaluate.py --question "What are germs?"
```

### Step 6: Launch Chatbot UI (Optional)

```bash
# Start the Streamlit chatbot interface
uv run streamlit run app.py
```

This opens a web-based chat interface where you can:
- Select any OCR + Chunking combination from dropdowns
- Choose prompt strategy (basic, cot, few_shot)
- Enable VLM enhancement for figure/diagram understanding
- Chat with the book and see supporting chunks + page images

### Quick Test (Full Pipeline)

```bash
# One-liner to verify everything works
uv sync && uv run scripts/setup.py && uv run scripts/answer.py --question "What are germs?"
```

---

## Chatbot UI (Streamlit)

The `app.py` provides a web-based chat interface for the RAG pipeline.

**Features:**
-  Chat interface for natural Q&A
-  Select any of the 9 OCR × Chunking combinations
-  Choose prompt strategy (basic / cot / few_shot)
-  VLM Enhancement toggle for figure understanding
-  View supporting chunks with page thumbnails
-  GPT-4o analyzes diagrams when VLM is enabled

**Launch:**
```bash
uv run streamlit run app.py
```

**VLM Enhancement:**
When enabled, the chatbot will:
1. Retrieve relevant text chunks (normal RAG)
2. Identify pages that might contain figures
3. Use GPT-4o-mini to analyze those page images
4. Include figure descriptions in the response

Bonus: You can now query for description of any images found in the book -- try "can you tell me the organs of the body from the book". The diagram is shown on page 10 of the pdf

**Note:** VLM adds latency (~2-5s) and API cost. Use for figure-related questions.

---

## Scripts Reference

### 1. `setup.py` — Build Vector Database

Processes the PDF through OCR → Chunking → Embedding → ChromaDB.

**Usage:**
```bash
uv run scripts/setup.py              # Best config only (tesseract + semantic)
uv run scripts/setup.py --all        # All 9 combinations (~30 min)
uv run scripts/setup.py --ocr tesseract                    # All chunking for one OCR
uv run scripts/setup.py --ocr tesseract --chunking page    # Specific combination
```

**Flags:**
| Flag | Values | Description |
|------|--------|-------------|
| `--all` | - | Setup all 9 OCR×Chunking combinations |
| `--ocr` | `tesseract`, `easyocr`, `paddleocr` | OCR model to use |
| `--chunking` | `page`, `parent_child`, `semantic` | Chunking strategy (requires `--ocr`) |

**Output:** Vector collections stored in `data/chroma_db/`

---

### 2. `answer.py` — Answer Questions (No Evaluation)

Answers questions using the RAG pipeline. Returns JSON with answer and supporting chunks.

**Usage:**
```bash
uv run scripts/answer.py --question "What are germs?"
uv run scripts/answer.py --all                              # Answer 3 target questions
uv run scripts/answer.py --all --output results/answers.json
```

**Flags:**
| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--question`, `-q` | string | - | Custom question to answer |
| `--all` | - | - | Answer all 3 target questions |
| `--ocr` | `tesseract`, `easyocr`, `paddleocr` | `tesseract` | OCR model |
| `--chunking` | `page`, `parent_child`, `semantic` | `semantic` | Chunking strategy |
| `--prompt` | `basic`, `cot`, `few_shot` | `basic` | Prompt strategy (see below) |
| `--output`, `-o` | filepath | - | Save JSON to file |

**Output Format:**
```json
{
  "question": "What are germs?",
  "answer": "According to the book...",
  "supporting_chunks": [
    {"chunk_id": "semantic_0017", "page": 8}
  ]
}
```

**Output Location:**
- Without `--output`: Prints JSON to stdout (CLI)
- With `--output`: Saves to specified file (e.g., `results/answers.json`)

---

### 3. `evaluate.py` — Answer + Evaluation Metrics

Same as `answer.py` but includes quality scores (groundedness, faithfulness, relevance).

**Usage:**
```bash
uv run scripts/evaluate.py                                  # using the 3 target questions as baseline
uv run scripts/evaluate.py --question "What are germs?"     # Evaluate custom question
uv run scripts/evaluate.py --eval-method all                # Use all metrics
uv run scripts/evaluate.py --output results/evaluation.json
```

**Flags:**
| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--question`, `-q` | string | - | Custom question (default: 3 target questions) |
| `--ocr` | `tesseract`, `easyocr`, `paddleocr` | `tesseract` | OCR model |
| `--chunking` | `page`, `parent_child`, `semantic` | `semantic` | Chunking strategy |
| `--prompt` | `basic`, `cot`, `few_shot` | `basic` | Prompt strategy (see below) |
| `--eval-method` | `simple`, `groundedness`, `ragas`, `all` | `groundedness` | Evaluation method |
| `--output`, `-o` | filepath | - | Save JSON to file |

**Evaluation Methods:**
| Method | Cost | Metrics |
|--------|------|---------|
| `simple` | Free | Confidence only |
| `groundedness` | $ | Groundedness (LLM-as-judge) |
| `ragas` | $$ | Faithfulness + Relevance |
| `all` | $$$ | All metrics |

**Prompt Strategies:**
| Strategy | Description | Best For |
|----------|-------------|----------|
| `basic` | Simple grounding instructions | General questions |
| `cot` | Chain-of-Thought: step-by-step reasoning | Complex multi-part questions |
| `few_shot` | Includes example Q&A pair | Consistent answer formatting |

**Output Location:**
- Always prints summary table to CLI
- With `--output`: Also saves full JSON to file

---

### 4. `full_evaluation.py` — Compare All Configurations

Runs the full experiment matrix: 3 OCR × 3 chunking = 9 combinations.
Evaluates each with all metrics and ranks by groundedness score.
Using the the 3 target questions as baseline here as well       

**Usage:**
```bash
uv run scripts/full_evaluation.py
```

**Output:**
- CLI: Progress updates + final ranking table
- File: `results/experiment_results.json`

---

## Target Questions

The assignment specifies these 3 benchmark questions:

1. What are the main ways to fight disease germs according to the book?
2. How does the book describe the importance of pure air and its effect on health?
3. Based on the principles described in the book, explain why preventing germs from entering the body and maintaining a clean environment together are more effective than either measure alone in reducing disease.

---

## Overall Approach and System Design

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌────────────────┐   │
│  │   PDF    │───▶│    OCR    │───▶│ Chunking  │───▶│   Embedding    │   │
│  │ (scanned)│    │           │    │           │    │                │   │
│  └──────────┘    └───────────┘    └───────────┘    └───────┬────────┘   │
│                   │ Tesseract      │ Page                   │            │
│                   │ EasyOCR        │ Parent-Child           ▼            │
│                   │ PaddleOCR      │ Semantic        ┌────────────┐     │
│                                                      │  ChromaDB  │     │
│                                                      │ VectorStore│     │
│                                                      └─────┬──────┘     │
│                                                            │            │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐           │            │
│  │  Answer  │◀───│ Generator │◀───│ Retriever │◀──────────┘            │
│  │  + Eval  │    │ (GPT-4o)  │    │  (Dense)  │                        │
│  └──────────┘    └───────────┘    └───────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Options | Description |
|-------|---------|-------------|
| **OCR** | Tesseract, EasyOCR, PaddleOCR | Extract text from scanned PDF pages. Tesseract uses grayscale + Otsu binarization preprocessing. |
| **Chunking** | Page, Parent-Child, Semantic | Split text into retrievable chunks |
| **Embedding** | BAAI/bge-small-en-v1.5 | Generate 384-dim dense vectors |
| **VectorStore** | ChromaDB | Persistent storage with cosine similarity |
| **Retrieval** | Dense, Parent-Child | Top-k similarity search |
| **Generation** | GPT-4o-mini | Generate grounded answers with citations |
| **Evaluation** | Groundedness, Ragas | LLM-as-judge + Ragas metrics |

### Design Patterns

- **Factory Pattern** - Components created via registry (`src/factory.py`)
- **Lazy Initialization** - Components loaded only when needed
- **Caching** - OCR results and chunks cached to disk for faster re-runs
- **Modular Design** - Mix any OCR × Chunking combination (9 total configurations)

### Experiment Matrix

We compare 3 OCR models × 3 chunking strategies = 9 combinations:

| OCR Model | Preprocessing | Chunking Strategies |
|-----------|---------------|---------------------|
| Tesseract | Grayscale + Otsu binarization | Page, Parent-Child, Semantic |
| EasyOCR | None (deep learning handles it) | Page, Parent-Child, Semantic |
| PaddleOCR | None (deep learning handles it) | Page, Parent-Child, Semantic |

---

## OCR Methods and Preprocessing

### PDF to Image Conversion

Before OCR, the scanned PDF is converted to PNG images:
- **Library**: PyMuPDF (fitz)
- **Resolution**: 300 DPI
- **Output**: One PNG per page (`data/pages/page_XXXX.png`)

### OCR Models

#### 1. Tesseract OCR
- **Type**: Traditional OCR engine
- **Preprocessing**: Grayscale + Otsu binarization (adaptive thresholding)
- **Why preprocessing?** Tesseract works best on clean black/white images

```python
# Preprocessing applied (src/ocr/tesseract_ocr.py)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

#### 2. EasyOCR
- **Type**: Deep learning-based (CRNN architecture)
- **Preprocessing**: None
- **Why no preprocessing?** Neural networks are trained on diverse images and handle variations internally

#### 3. PaddleOCR
- **Type**: Deep learning-based (PP-OCR)
- **Preprocessing**: None
- **Features**: Built-in text detection + angle classification for rotated text
- **Note**: Requires `paddlepaddle==2.6.2` on Windows

### Text Post-Processing

After OCR, text is cleaned before chunking (`src/ocr/preprocessing.py`):

| Step | What it does | Example |
|------|--------------|---------|
| `rejoin_hyphenated_words` | Fix line-break hyphens | `preven-\ntion` → `prevention` |
| `collapse_whitespace` | Normalize spacing | `word    word` → `word word` |
| `strip_page_numbers` | Remove standalone numbers | `\n42\n` → removed |
| `rejoin_broken_lines` | Merge incomplete sentences | Lines not ending in `.!?` joined |

---

## Chunking Strategies

After OCR extracts text, it must be split into retrievable chunks. We compare three strategies:

### 1. Page Chunking
- **Method**: Each PDF page becomes one chunk. Simple 1:1 mapping.
- **Implementation**: `src/chunking/page_chunker.py`

| Pros | Cons |
|------|------|
| Fast and simple | Page boundaries may split related content |
| Preserves document structure | Large chunks (~500-1000 tokens) may dilute relevance |
| Easy to cite page numbers | No semantic awareness |

### 2. Parent-Child Chunking
- **Method**: Decouples search from retrieval by indexing small "Child" chunks for search accuracy but retrieving their larger "Parent" chunks to provide full context to the LLM.
- **Implementation**: `src/chunking/parent_child_chunker.py`

| Component | Size | Purpose |
|-----------|------|---------|
| Parent | ~1000 chars | Provides context to LLM |
| Child | ~200 chars | Used for similarity search |

| Pros | Cons |
|------|------|
| Best of both worlds: precise search + rich context | Requires storing both parent and child chunks |
| Small chunks match specific queries better | More complex lineage tracking |
| Parent provides sufficient context for generation | ~2x storage overhead |

### 3. Semantic Chunking
- **Method**: Splits text into sentences, embeds each one, and groups adjacent sentences with high cosine similarity. When similarity drops below threshold, a new chunk begins.
- **Implementation**: `src/chunking/semantic_chunker.py`

| Pros | Cons |
|------|------|
| High contextual integrity | Computationally expensive (requires embedding during indexing) |
| Chunks represent complete thoughts/topics | Variable chunk sizes |
| Better retrieval quality for concept-based queries | May create very large chunks if topic spans many pages |

---

## Embedding Strategy

### Model: BAAI/bge-small-en-v1.5
**Why this model?**
- Good balance of quality vs speed
- Small enough to run locally without GPU
- Optimized for retrieval tasks (trained with contrastive learning)
- Outperforms larger models on MTEB retrieval benchmarks for its size

### Vector Storage: ChromaDB

| Property | Value |
|----------|-------|
| **Database** | ChromaDB (persistent) |
| **Distance Metric** | Cosine similarity |
| **Storage Path** | `data/chroma_db/` |
| **Collection Naming** | `{ocr}__{chunking}` (e.g., `tesseract__semantic`) |

Each OCR × Chunking combination gets its own collection, enabling fair comparison.

---

## Retrieval and LLM Prompt Construction

### Retrieval

- **Method**: Dense retrieval with top-k similarity search
- **k**: 5 chunks retrieved per query
- **Reranking**: None (pure embedding similarity)

For parent-child chunking, child chunks are searched but parent chunks are returned.

### Prompt Construction

The pipeline supports **3 configurable prompt strategies** via `--prompt` flag:

#### Strategy: `basic` (default)
Simple grounding instructions:
```
System: You are a helpful assistant... ground every claim in context...
        Do not use any outside knowledge.

User:   Answer using ONLY the provided context chunks.
        Question: {question}
        Context: {context}
        Instructions: Answer clearly, reference chunks.
```

#### Strategy: `cot` (Chain-of-Thought)
Step-by-step reasoning before answering:
```
System: ...Think step-by-step before providing your final answer.

User:   Think through this step-by-step:
        1. Identify Relevant Chunks: Which chunks contain relevant info?
        2. Extract Key Facts: What are the main points?
        3. Synthesize Answer: Combine facts to answer the question.
```

#### Strategy: `few_shot`
Includes example Q&A to guide format:
```
User:   Here is an example of how to answer:
        ---
        Example Question: What causes tuberculosis?
        Example Context: [Chunk 1] Tuberculosis is caused by...
        Example Answer: According to the book, tuberculosis is caused by...
        ---
        Now answer: {question}
```

### Generation

| Property | Value |
|----------|-------|
| **Model** | GPT-4o-mini |
| **Temperature** | 0 (deterministic) |
| **Max Tokens** | 1024 |

The LLM receives the question and retrieved chunks, then generates a grounded answer with citations.

---

## Evaluation Metrics

We evaluate answer quality using multiple metrics:

| Metric | Method | Description |
|--------|--------|-------------|
| **Groundedness** | LLM-as-judge | Does the answer stay grounded in the retrieved context? (0-1) |
| **Faithfulness** | Ragas | Are answer claims supported by the context? (0-1) |
| **Relevance** | Ragas | Is the answer relevant to the question? (0-1) |
| **Confidence** | Retrieval score | Average cosine similarity of retrieved chunks |

---

## Assumptions, Limitations & Known Failure Cases

### Assumptions

| Assumption | Description |
|------------|-------------|
| **Scanned PDF** | The input PDF is image-based (no selectable text). Text-based PDFs would skip OCR. |
| **English text** | OCR models and embeddings are optimized for English. Other languages may have degraded accuracy. |

### Limitations

| Limitation | Impact | 
|------------|--------|
| **OCR errors** | Old/degraded scans may have character recognition errors (e.g., "rn" → "m") | 
| **No table/figure extraction** | Tables and diagrams are converted to text, losing structure | 
| **Context window limits** | Very long chunks may exceed embedding model's 512 token limit, so theres a change semantic chunking may not be impactful for some sections |
| **No image understanding** | Questions about figures, charts, or illustrations will fail with CLI scripts. Use Streamlit app with VLM Enhancement enabled for figure analysis. |

### Known Failure Cases

| Failure Case | Example | Why It Fails |
|--------------|---------|--------------|
| **Specific page references** | "What is on page 47?" | Page chunking loses exact boundaries; semantic chunking ignores pages |
| **Figure/table questions** | "Describe the diagram on page 23" | OCR only extracts text. Enable VLM Enhancement in Streamlit app for figure analysis. |
| **OCR-corrupted terms** | Questions with proper nouns that OCR misread | Embedding won't match corrupted text |
| **Very long answers needed** | Complex questions requiring 1000+ word answers | Max tokens limit (1024) may truncate; use multiple questions |
| **Negation/contrast** | "What does the book NOT recommend?" | Retrieval optimizes for similarity, not negation |



## Project Structure

```
├── app.py                      # Streamlit chatbot UI
├── config/
│   └── default.yaml            # Pipeline configuration (models, paths, parameters)
├── data/
│   ├── pages/                  # Extracted PNG images from PDF
│   ├── ocr_output/             # Cached OCR results per model
│   ├── chunks/                 # Cached chunk data
│   └── chroma_db/              # Vector database collections
├── results/                    # Output files from evaluations
│   └── experiment_results.json # Full experiment comparison
├── scripts/
│   ├── setup.py                # Build vector database
│   ├── answer.py               # Answer questions (no eval)
│   ├── evaluate.py             # Answer + evaluation metrics
│   └── full_evaluation.py      # Compare all 9 configurations
├── src/
│   ├── ocr/                    # OCR extractors
│   │   ├── tesseract_ocr.py    # Tesseract with preprocessing
│   │   ├── easyocr_extractor.py
│   │   ├── paddleocr_extractor.py
│   │   └── vlm_extractor.py    # Vision-Language Model (for future enhancement)
│   ├── chunking/               # Chunking strategies
│   │   ├── page_chunker.py
│   │   ├── parent_child_chunker.py
│   │   └── semantic_chunker.py
│   ├── embedding/              # Embedding models
│   │   └── sentence_transformer.py  # BAAI/bge-small-en-v1.5
│   ├── vectorstore/            # Vector database
│   │   └── chroma_store.py
│   ├── retrieval/              # Retrieval strategies
│   │   ├── dense_retriever.py
│   │   └── parent_child_retriever.py
│   ├── generation/             # LLM generation
│   │   ├── openai_generator.py
│   │   └── prompt_templates.py # Configurable prompt strategies
│   ├── evaluation/             # Evaluation metrics
│   │   ├── evaluator.py
│   │   ├── groundedness.py     # LLM-as-judge
│   │   ├── ragas_evaluator.py  # Faithfulness + Relevance
│   │   └── langfuse_tracker.py # Optional observability
│   ├── pipeline.py             # Main RAG pipeline orchestration
│   ├── factory.py              # Component factory with lazy loading
│   ├── config.py               # Configuration loader
│   └── models.py               # Pydantic data models
├── GenAI_Assignment_Health_v1/
│   └── Principles of Public Health.pdf  # Source document
├── pyproject.toml              # Project dependencies (uv/pip)
└── README.md
```
