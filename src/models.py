"""Pydantic models for the RAG pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PageImage(BaseModel):
    page_number: int
    image_path: str


class OCRResult(BaseModel):
    page_number: int
    text: str
    model_name: str


class TextChunk(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    metadata: dict = Field(default_factory=dict)


class SupportingChunk(BaseModel):
    chunk_id: str
    page: int


class RAGResponse(BaseModel):
    question: str
    answer: str
    supporting_chunks: list[SupportingChunk]


class RetrievedChunk(BaseModel):
    chunk: TextChunk
    score: float


class EvaluationScore(BaseModel):
    groundedness: float = 0.0
    relevance: float = 0.0
    faithfulness: float = 0.0
    confidence: float | None = None


class EvaluatedResponse(BaseModel):
    response: RAGResponse
    evaluation: EvaluationScore
