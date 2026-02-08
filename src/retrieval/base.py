"""Abstract base class for retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import RetrievedChunk


class BaseRetriever(ABC):
    """All retrievers must subclass this."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query."""
