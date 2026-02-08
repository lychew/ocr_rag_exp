"""Dense retriever — embeds query, searches vector store."""

from __future__ import annotations

from src.embedding.base import BaseEmbedding
from src.models import RetrievedChunk
from src.retrieval.base import BaseRetriever
from src.vectorstore.base import BaseVectorStore


class DenseRetriever(BaseRetriever):

    def __init__(
        self,
        embedding: BaseEmbedding,
        vectorstore: BaseVectorStore,
        **kwargs,
    ):
        self.embedding = embedding
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vec = self.embedding.embed_single(query)
        return self.vectorstore.query(query_vec, top_k=top_k)
