"""RAG Pipeline orchestrator — wires all components together."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from src.config import load_config
from src.models import OCRResult, RAGResponse, TextChunk, EvaluatedResponse, RetrievedChunk
from src.ocr.preprocessing import clean_ocr_text, extract_pages_from_pdf

console = Console()


class RAGPipeline:
    """End-to-end pipeline: ingest (OCR→chunk→embed→store) and query (retrieve→generate)."""

    def __init__(
        self,
        ocr_name: str = "tesseract",
        chunking_name: str = "page",
        prompt_strategy: str = "basic",
    ):
        self.cfg = load_config(ocr_name=ocr_name, chunking_name=chunking_name)
        self.ocr_name = ocr_name
        self.chunking_name = chunking_name
        self.prompt_strategy = prompt_strategy
        self.collection_name = f"{ocr_name}__{chunking_name}"

        # Lazy-initialised components
        self._ocr = None
        self._chunker = None
        self._embedding = None
        self._vectorstore = None
        self._retriever = None
        self._generator = None

    # --- Component accessors (lazy init) ---

    @property
    def ocr(self):
        if self._ocr is None:
            from src import factory
            self._ocr = factory.create("ocr", self.ocr_name)
        return self._ocr

    @property
    def chunker(self):
        if self._chunker is None:
            from src import factory
            self._chunker = factory.create("chunking", self.chunking_name)
        return self._chunker

    @property
    def embedding(self):
        if self._embedding is None:
            from src import factory
            self._embedding = factory.create(
                "embedding",
                "sentence_transformer",
                model_name=self.cfg["embedding"]["model_name"],
            )
        return self._embedding

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            from src import factory
            self._vectorstore = factory.create(
                "vectorstore",
                "chroma",
                collection_name=self.collection_name,
                persist_dir=self.cfg["vectorstore"]["persist_dir"],
            )
        return self._vectorstore

    @property
    def retriever(self):
        if self._retriever is None:
            from src import factory
            # Use parent-child retriever for parent-child chunking
            retriever_type = "parent_child" if self.chunking_name == "parent_child" else "dense"
            self._retriever = factory.create(
                "retrieval",
                retriever_type,
                embedding=self.embedding,
                vectorstore=self.vectorstore,
            )
        return self._retriever

    @property
    def generator(self):
        if self._generator is None:
            from src import factory
            gen_cfg = self.cfg["generation"]
            self._generator = factory.create(
                "generation",
                "openai",
                model_name=gen_cfg["model_name"],
                temperature=gen_cfg["temperature"],
                max_tokens=gen_cfg["max_tokens"],
                prompt_strategy=self.prompt_strategy,
            )
        return self._generator

    # --- Ingest ---

    def ingest(self) -> int:
        """Run the full ingest pipeline. Returns number of chunks indexed."""
        pdf_path = self.cfg["pdf_path"]
        data_dir = self.cfg["data_dir"]
        dpi = self.cfg["ocr"].get("dpi", 300)

        # 1. PDF → page images
        console.print(f"[bold]Extracting pages from PDF...[/bold]")
        pages_dir = str(Path(data_dir) / "pages")
        image_paths = extract_pages_from_pdf(pdf_path, pages_dir, dpi=dpi)
        console.print(f"  {len(image_paths)} page images ready")

        # 2. OCR
        ocr_cache_dir = Path(data_dir) / "ocr_output" / self.ocr_name
        ocr_results = self._run_ocr(image_paths, ocr_cache_dir)
        console.print(f"  {len(ocr_results)} pages OCR'd with {self.ocr_name}")

        # 3. Chunk
        chunks = self.chunker.chunk(ocr_results)
        self._save_chunks(chunks)
        console.print(f"  {len(chunks)} chunks created with {self.chunking_name} strategy")

        # 4. Embed + store
        console.print(f"[bold]Embedding and indexing...[/bold]")
        texts = [c.text for c in chunks]
        embeddings = self.embedding.embed(texts)
        self.vectorstore.add_chunks(chunks, embeddings)
        console.print(
            f"  {self.vectorstore.count()} chunks indexed in collection "
            f"[cyan]{self.collection_name}[/cyan]"
        )

        return len(chunks)

    def _run_ocr(self, image_paths: list[str], cache_dir: Path) -> list[OCRResult]:
        """Run OCR with caching."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "ocr_results.json"

        if cache_file.exists():
            console.print(f"  Loading cached OCR results from {cache_file}")
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return [OCRResult(**r) for r in raw]

        results = self.ocr.extract_pages(image_paths)

        # Post-process
        for r in results:
            r.text = clean_ocr_text(r.text)

        # Cache
        cache_file.write_text(
            json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return results

    def _save_chunks(self, chunks: list[TextChunk]) -> None:
        """Cache chunks to disk."""
        data_dir = self.cfg["data_dir"]
        chunk_dir = Path(data_dir) / "chunks" / self.collection_name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out_file = chunk_dir / "chunks.json"
        out_file.write_text(
            json.dumps([c.model_dump() for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Query ---

    def answer_question(
        self, question: str
    ) -> tuple[RAGResponse, list[RetrievedChunk]]:
        """Retrieve relevant chunks and generate an answer.

        Returns:
            Tuple of (RAGResponse, list of RetrievedChunk used)
        """
        top_k = self.cfg["retrieval"]["top_k"]
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        response = self.generator.generate(question, retrieved)
        return response, retrieved

    def answer_all_questions(
        self, with_evaluation: bool = False, eval_method: str = "simple"
    ) -> list[RAGResponse] | list[EvaluatedResponse]:
        """Answer all target questions from config.

        Args:
            with_evaluation: If True, run evaluation and return EvaluatedResponse
            eval_method: Evaluation method ("simple", "groundedness", "ragas", "all")

        Returns:
            List of RAGResponse (if with_evaluation=False) or EvaluatedResponse
        """
        questions = self.cfg.get("questions", [])
        responses = []

        evaluator = None
        if with_evaluation:
            from src.evaluation.evaluator import Evaluator, EvaluatorConfig
            evaluator = Evaluator(EvaluatorConfig(method=eval_method))

        for q in questions:
            console.print(f"\n[bold yellow]Q:[/bold yellow] {q}")
            resp, retrieved = self.answer_question(q)
            console.print(f"[bold green]A:[/bold green] {resp.answer[:200]}...")

            if with_evaluation and evaluator:
                evaluated = evaluator.evaluate(resp, retrieved)
                console.print(
                    f"[bold cyan]Scores:[/bold cyan] "
                    f"faith={evaluated.evaluation.faithfulness:.2f} "
                    f"rel={evaluated.evaluation.relevance:.2f} "
                    f"ground={evaluated.evaluation.groundedness:.2f} "
                    f"conf={evaluated.evaluation.confidence:.2f}"
                )
                responses.append(evaluated)
            else:
                responses.append(resp)

        # Log to Langfuse if evaluation was run
        if with_evaluation and responses:
            from src.evaluation.langfuse_tracker import get_tracker
            tracker = get_tracker()
            tracker.log_full_evaluation(
                ocr_model=self.ocr_name,
                chunking_strategy=self.chunking_name,
                evaluated_responses=responses,
            )

        return responses
