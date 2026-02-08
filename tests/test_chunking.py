"""Tests for chunking strategies."""

from src.chunking.page_chunker import PageChunker
from src.models import OCRResult


def test_page_chunker(sample_ocr_results):
    chunker = PageChunker()
    chunks = chunker.chunk(sample_ocr_results)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "page_0001"
    assert chunks[0].page_number == 1
    assert "disease germs" in chunks[0].text


def test_page_chunker_skips_empty():
    ocr_results = [
        OCRResult(page_number=1, text="content", model_name="test"),
        OCRResult(page_number=2, text="   ", model_name="test"),
        OCRResult(page_number=3, text="more content", model_name="test"),
    ]
    chunker = PageChunker()
    chunks = chunker.chunk(ocr_results)
    assert len(chunks) == 2
