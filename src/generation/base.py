"""Abstract base class for LLM generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import RAGResponse, RetrievedChunk


class BaseGenerator(ABC):
    """All LLM generators must subclass this."""

    @abstractmethod
    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> RAGResponse:
        """Generate an answer grounded in the retrieved chunks."""
