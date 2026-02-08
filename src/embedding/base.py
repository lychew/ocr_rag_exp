"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """All embedding providers must subclass this."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts."""

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
