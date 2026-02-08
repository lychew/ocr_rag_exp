"""OCR post-processing — clean raw OCR text before chunking."""

from __future__ import annotations

import re


def clean_ocr_text(text: str) -> str:
    """Apply all cleaning steps to raw OCR output."""
    text = rejoin_hyphenated_words(text)
    text = collapse_whitespace(text)
    text = strip_page_numbers(text)
    text = rejoin_broken_lines(text)
    return text.strip()


def rejoin_hyphenated_words(text: str) -> str:
    """Rejoin words split across lines with a hyphen: 'preven-\\ntion' → 'prevention'."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to a single space."""
    return re.sub(r"[^\S\n]+", " ", text)


def strip_page_numbers(text: str) -> str:
    """Remove isolated page numbers (standalone lines that are just digits)."""
    return re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)


def rejoin_broken_lines(text: str) -> str:
    """Join lines that don't end with sentence-ending punctuation."""
    lines = text.split("\n")
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        if merged and merged[-1] and not re.search(r"[.!?:;]\s*$", merged[-1]):
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)
    return "\n".join(merged)


def extract_pages_from_pdf(pdf_path: str, output_dir: str, dpi: int = 300) -> list[str]:
    """Convert PDF pages to images using PyMuPDF. Returns list of image file paths."""
    from pathlib import Path
    import fitz  # PyMuPDF

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing = sorted(out.glob("page_*.png"))
    if existing:
        return [str(p) for p in existing]

    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)
    paths = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat)
        p = out / f"page_{i:04d}.png"
        pix.save(str(p))
        paths.append(str(p))
    doc.close()
    return paths
