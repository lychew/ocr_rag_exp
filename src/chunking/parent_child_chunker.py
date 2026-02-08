"""Parent-child chunking strategy.

Creates a hierarchy:
- Parent chunks: 800-1500 tokens, provide context for LLM
- Child chunks: 200-400 tokens with overlap, used for precise retrieval

During retrieval, we search children but return parents.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from src.chunking.base import BaseChunker
from src.models import OCRResult, TextChunk


@dataclass
class ChunkConfig:
    parent_chunk_size: int = 1200  
    child_chunk_size: int = 300   
    child_overlap: int = 50      
    chars_per_token: float = 4.0


class ParentChildChunker(BaseChunker):
    name = "parent_child"

    def __init__(
        self,
        parent_chunk_size: int = 1200,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
        **kwargs,
    ):
        self.config = ChunkConfig(
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
        )

    def chunk(self, ocr_results: list[OCRResult]) -> list[TextChunk]:
        """Create parent and child chunks from OCR results.

        Returns all chunks (both parents and children).
        Children have metadata pointing to their parent.
        """
        full_text = ""
        page_boundaries = [] 

        for ocr in ocr_results:
            text = ocr.text.strip()
            if not text:
                continue
            page_boundaries.append((len(full_text), ocr.page_number))
            full_text += text + "\n\n"

        if not full_text.strip():
            return []

        parent_char_size = int(self.config.parent_chunk_size * self.config.chars_per_token)
        parents = self._split_into_chunks(
            full_text,
            chunk_size=parent_char_size,
            overlap=0,
            page_boundaries=page_boundaries,
        )

        all_chunks = []

        for i, (parent_text, parent_pages) in enumerate(parents):
            parent_id = f"parent_{i:04d}"
            pages_str = ",".join(str(p) for p in parent_pages)
            parent_chunk = TextChunk(
                chunk_id=parent_id,
                text=parent_text,
                page_number=parent_pages[0] if parent_pages else 0,
                metadata={
                    "strategy": self.name,
                    "chunk_type": "parent",
                    "pages": pages_str,
                },
            )
            all_chunks.append(parent_chunk)

            child_char_size = int(self.config.child_chunk_size * self.config.chars_per_token)
            child_overlap = int(self.config.child_overlap * self.config.chars_per_token)

            children = self._split_text_with_overlap(
                parent_text,
                chunk_size=child_char_size,
                overlap=child_overlap,
            )

            for j, child_text in enumerate(children):
                child_id = f"child_{i:04d}_{j:03d}"
                child_chunk = TextChunk(
                    chunk_id=child_id,
                    text=child_text,
                    page_number=parent_pages[0] if parent_pages else 0,
                    metadata={
                        "strategy": self.name,
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "pages": pages_str,
                    },
                )
                all_chunks.append(child_chunk)

        return all_chunks

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        page_boundaries: list[tuple[int, int]],
    ) -> list[tuple[str, list[int]]]:
        """Split text into chunks, tracking which pages each chunk spans."""
        chunks = []

        paragraphs = re.split(r"\n\n+", text)

        current_chunk = ""
        current_pages = set()
        char_position = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                char_position += 2 
                continue

            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append((current_chunk.strip(), sorted(current_pages)))
                current_chunk = ""
                current_pages = set()

            page = self._find_page_for_position(char_position, page_boundaries)
            current_pages.add(page)

            current_chunk += para + "\n\n"
            char_position += len(para) + 2

        if current_chunk.strip():
            chunks.append((current_chunk.strip(), sorted(current_pages)))

        return chunks

    def _split_text_with_overlap(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                search_start = end - int(chunk_size * 0.2)
                sentence_end = text.rfind(". ", search_start, end)
                if sentence_end > search_start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(text) - overlap:
                break

        return chunks

    def _find_page_for_position(
        self,
        position: int,
        page_boundaries: list[tuple[int, int]],
    ) -> int:
        """Find which page a character position belongs to."""
        page = 1
        for boundary_pos, page_num in page_boundaries:
            if position >= boundary_pos:
                page = page_num
            else:
                break
        return page
