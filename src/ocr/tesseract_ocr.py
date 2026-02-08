"""Tesseract OCR engine with image preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from src.ocr.base import BaseOCR

# Standard Windows install path
_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractOCR(BaseOCR):
    name = "tesseract"

    def __init__(self, lang: str = "eng", **kwargs):
        self.lang = lang
        if Path(_TESSERACT_WIN).exists():
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN

    def _preprocess_image(self, image_path: str | Path) -> np.ndarray:
        """Apply grayscale + Otsu binarization for better OCR accuracy."""
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def extract_page(self, image_path: str | Path) -> str:
        processed = self._preprocess_image(image_path)
        img = Image.fromarray(processed)

        text = pytesseract.image_to_string(img, lang=self.lang)
        return text
