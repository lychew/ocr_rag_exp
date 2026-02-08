"""EasyOCR engine."""

from __future__ import annotations

from pathlib import Path

import easyocr

from src.ocr.base import BaseOCR


class EasyOCR(BaseOCR):
    name = "easyocr"

    def __init__(self, lang: list[str] | None = None, gpu: bool = True, **kwargs):
        self.lang = lang or ["en"]
        self.reader = easyocr.Reader(self.lang, gpu=gpu)

    def extract_page(self, image_path: str | Path) -> str:
        results = self.reader.readtext(str(image_path), detail=0, paragraph=True)
        return "\n".join(results)
