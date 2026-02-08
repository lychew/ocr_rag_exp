"""Parent-child retriever — search children, return parents.

This retriever is designed for the parent-child chunking strategy:
1. Search child chunks (more precise matches)
2. Return their parent chunks (more context for LLM)
"""

from __future__ import annotations

from src.embedding.base import BaseEmbedding
from src.models import RetrievedChunk, TextChunk
from src.retrieval.base import BaseRetriever
from src.vectorstore.base import BaseVectorStore


class ParentChildRetriever(BaseRetriever):

    def __init__(
        self,
        embedding: BaseEmbedding,
        vectorstore: BaseVectorStore,
        **kwargs,
    ):
        self.embedding = embedding
        self.vectorstore = vectorstore
        # Cache parent chunks for fast lookup
        self._parent_cache: dict[str, TextChunk] = {}
        self._cache_built = False

    def _build_parent_cache(self) -> None:
        """Build a cache of parent_id -> parent_chunk."""
        if self._cache_built:
            return

        # Query all chunks to find parents
        # This is a bit hacky but ChromaDB doesn't have a "get all" that's efficient
        # We'll rely on the chunks being stored with metadata

        # For now, we'll build the cache lazily as we encounter parents
        self._cache_built = True

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve relevant chunks.

        For parent-child strategy:
        - Search ALL chunks (children are more precise)
        - If we find a child, return its parent instead
        - Deduplicate by parent_id
        """
        query_vec = self.embedding.embed_single(query)

        # Get more results than needed since we'll deduplicate by parent
        raw_results = self.vectorstore.query(query_vec, top_k=top_k * 3)

        # Track seen parents to avoid duplicates
        seen_parents: set[str] = set()
        final_results: list[RetrievedChunk] = []

        for result in raw_results:
            chunk = result.chunk
            meta = chunk.metadata

            # Check if this is a child chunk
            if meta.get("chunk_type") == "child":
                parent_id = meta.get("parent_id")
                if parent_id and parent_id not in seen_parents:
                    # Try to get the parent chunk
                    parent = self._get_parent_chunk(parent_id)
                    if parent:
                        seen_parents.add(parent_id)
                        final_results.append(
                            RetrievedChunk(chunk=parent, score=result.score)
                        )
            else:
                # It's a parent chunk or regular chunk
                chunk_id = chunk.chunk_id
                if chunk_id not in seen_parents:
                    seen_parents.add(chunk_id)
                    final_results.append(result)

            if len(final_results) >= top_k:
                break

        return final_results[:top_k]

    def _get_parent_chunk(self, parent_id: str) -> TextChunk | None:
        """Get a parent chunk by ID."""
        # Check cache first
        if parent_id in self._parent_cache:
            return self._parent_cache[parent_id]

        # Query the vector store for this specific chunk
        # ChromaDB allows querying by ID
        try:
            results = self.vectorstore.collection.get(
                ids=[parent_id],
                include=["documents", "metadatas"],
            )

            if results["ids"] and results["ids"][0]:
                doc = results["documents"][0]
                meta = results["metadatas"][0]

                parent = TextChunk(
                    chunk_id=parent_id,
                    text=doc,
                    page_number=meta.get("page_number", 0),
                    metadata=meta,
                )
                self._parent_cache[parent_id] = parent
                return parent
        except Exception:
            pass

        return None
