"""Semantic chunking strategy.

Uses embedding similarity between sentences to find natural breakpoints.
Sentences with low similarity to their neighbors indicate topic changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.chunking.base import BaseChunker
from src.models import OCRResult, TextChunk


@dataclass
class SemanticConfig:
    min_chunk_size: int = 100     
    max_chunk_size: int = 2000    
    similarity_threshold: float = 0.5  
    embedding_model: str = "BAAI/bge-small-en-v1.5"


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        similarity_threshold: float = 0.5,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        **kwargs,
    ):
        self.config = SemanticConfig(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
        )
        self._embedder = None

    @property
    def embedder(self):
        """Lazy load embedder to avoid slow imports."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def chunk(self, ocr_results: list[OCRResult]) -> list[TextChunk]:
        """Create semantically coherent chunks from OCR results."""
        all_chunks = []
        chunk_counter = 0

        for ocr in ocr_results:
            text = ocr.text.strip()
            if not text:
                continue

            sentences = self._split_into_sentences(text)

            if len(sentences) <= 1:
                if text:
                    all_chunks.append(
                        TextChunk(
                            chunk_id=f"semantic_{chunk_counter:04d}",
                            text=text,
                            page_number=ocr.page_number,
                            metadata={"strategy": self.name},
                        )
                    )
                    chunk_counter += 1
                continue

            embeddings = self.embedder.encode(sentences, normalize_embeddings=True)
            breakpoints = self._find_breakpoints(embeddings, sentences)
            page_chunks = self._create_chunks_from_breakpoints(
                sentences, breakpoints, ocr.page_number, chunk_counter
            )

            all_chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

        return all_chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_breakpoints(
        self,
        embeddings: np.ndarray,
        sentences: list[str],
    ) -> list[int]:
        """Find sentence indices where topic changes occur."""
        if len(embeddings) < 2:
            return []

        breakpoints = []
        current_chunk_size = 0

        for i in range(len(embeddings) - 1):
            current_chunk_size += len(sentences[i])
            similarity = np.dot(embeddings[i], embeddings[i + 1])
            should_break = False


            if similarity < self.config.similarity_threshold:
                should_break = True


            if current_chunk_size >= self.config.max_chunk_size:
                should_break = True


            if current_chunk_size < self.config.min_chunk_size:
                should_break = False

            if should_break:
                breakpoints.append(i + 1) 
                current_chunk_size = 0

        return breakpoints

    def _create_chunks_from_breakpoints(
        self,
        sentences: list[str],
        breakpoints: list[int],
        page_number: int,
        start_counter: int,
    ) -> list[TextChunk]:
        """Create TextChunk objects from sentences and breakpoints."""
        chunks = []

        boundaries = [0] + breakpoints + [len(sentences)]

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            if chunk_text.strip():
                chunks.append(
                    TextChunk(
                        chunk_id=f"semantic_{start_counter + i:04d}",
                        text=chunk_text,
                        page_number=page_number,
                        metadata={"strategy": self.name},
                    )
                )

        return chunks
