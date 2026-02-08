"""ChromaDB vector store implementation."""

from __future__ import annotations

from pathlib import Path

import chromadb

from src.models import TextChunk, RetrievedChunk
from src.vectorstore.base import BaseVectorStore


class ChromaStore(BaseVectorStore):

    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: str = "data/vectordb",
        **kwargs,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"page_number": c.page_number, "chunk_id": c.chunk_id, **c.metadata}
                for c in chunks
            ],
        )

    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()) if self.count() > 0 else top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                doc = results["documents"][0][i]
                distance = results["distances"][0][i]
                # ChromaDB cosine distance: similarity = 1 - distance
                score = 1.0 - distance
                retrieved.append(
                    RetrievedChunk(
                        chunk=TextChunk(
                            chunk_id=chunk_id,
                            text=doc,
                            page_number=meta.get("page_number", 0),
                            metadata=meta,
                        ),
                        score=score,
                    )
                )
        return retrieved

    def count(self) -> int:
        return self.collection.count()
