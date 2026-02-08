"""Shared test fixtures."""

import pytest

from src.models import OCRResult, TextChunk, RetrievedChunk


@pytest.fixture
def sample_ocr_results():
    return [
        OCRResult(page_number=1, text="This is page one about disease germs.", model_name="tesseract"),
        OCRResult(page_number=2, text="Pure air is essential for health.", model_name="tesseract"),
        OCRResult(page_number=3, text="Preventing germs and clean environment work together.", model_name="tesseract"),
    ]


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(chunk_id="page_0001", text="This is page one about disease germs.", page_number=1),
        TextChunk(chunk_id="page_0002", text="Pure air is essential for health.", page_number=2),
        TextChunk(chunk_id="page_0003", text="Preventing germs and clean environment work together.", page_number=3),
    ]


@pytest.fixture
def sample_retrieved_chunks(sample_chunks):
    return [
        RetrievedChunk(chunk=sample_chunks[0], score=0.9),
        RetrievedChunk(chunk=sample_chunks[1], score=0.85),
    ]
