from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import OCRResult, TextChunk


class BaseChunker(ABC):
    """All chunking strategies must subclass this."""

    name: str = "base"

    @abstractmethod
    def chunk(self, ocr_results: list[OCRResult]) -> list[TextChunk]:
        """Split OCR results into text chunks."""
