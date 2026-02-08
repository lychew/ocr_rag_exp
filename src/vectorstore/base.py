"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import TextChunk, RetrievedChunk


class BaseVectorStore(ABC):
    """All vector store implementations must subclass this."""

    @abstractmethod
    def add_chunks(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> None:
        """Store chunks with their embeddings."""

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Find the top-k most similar chunks to the query embedding."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""
