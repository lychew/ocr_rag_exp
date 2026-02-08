"""PaddleOCR engine with built-in layout detection."""

from __future__ import annotations

import os
os.environ["FLAGS_use_mkldnn"] = "0"  # Disable OneDnn globally

from pathlib import Path

from paddleocr import PaddleOCR

from src.ocr.base import BaseOCR


class PaddleOCRExtractor(BaseOCR):
    name = "paddleocr"

    def __init__(self, lang: str = "en", use_gpu: bool = False, **kwargs):
        self.lang = lang
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang=self.lang,
            use_gpu=use_gpu,
            enable_mkldnn=False, 
            show_log=False,
        )

    def extract_page(self, image_path: str | Path) -> str:
        result = self.ocr.ocr(str(image_path), cls=True)

        if not result or not result[0]:
            return ""
        lines = []
        for item in result[0]:
            bbox, (text, conf) = item
            y_top = min(bbox[0][1], bbox[1][1])
            lines.append((y_top, text))

        lines.sort(key=lambda x: x[0])
        return "\n".join(text for _, text in lines)
