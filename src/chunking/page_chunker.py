
from __future__ import annotations

from src.chunking.base import BaseChunker
from src.models import OCRResult, TextChunk


class PageChunker(BaseChunker):
    name = "page"

    def __init__(self, **kwargs):
        pass

    def chunk(self, ocr_results: list[OCRResult]) -> list[TextChunk]:
        chunks = []
        for ocr in ocr_results:
            text = ocr.text.strip()
            if not text:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=f"page_{ocr.page_number:04d}",
                    text=text,
                    page_number=ocr.page_number,
                    metadata={"strategy": self.name},
                )
            )
        return chunks
