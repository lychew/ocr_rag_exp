"""Abstract base class for OCR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models import OCRResult


class BaseOCR(ABC):
    """All OCR implementations must subclass this."""

    name: str = "base"

    @abstractmethod
    def extract_page(self, image_path: str | Path) -> str:
        """Extract raw text from a single page image."""

    def extract_pages(
        self, image_paths: list[str | Path]
    ) -> list[OCRResult]:
        """Extract text from multiple page images."""
        results = []
        for i, img_path in enumerate(image_paths, start=1):
            raw_text = self.extract_page(img_path)
            results.append(
                OCRResult(page_number=i, text=raw_text, model_name=self.name)
            )
        return results
