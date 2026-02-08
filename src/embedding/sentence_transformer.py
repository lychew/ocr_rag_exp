"""Sentence-transformer embedding provider."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.embedding.base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
