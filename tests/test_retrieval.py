"""Tests for vector store and retrieval."""

import tempfile
import pytest

from src.embedding.sentence_transformer import SentenceTransformerEmbedding
from src.models import TextChunk
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.dense_retriever import DenseRetriever


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedding(model_name="BAAI/bge-small-en-v1.5")


@pytest.fixture
def store_with_data(embedder, sample_chunks, tmp_path):
    store = ChromaStore(
        collection_name="test_collection",
        persist_dir=str(tmp_path / "test_vectordb"),
    )
    embeddings = embedder.embed([c.text for c in sample_chunks])
    store.add_chunks(sample_chunks, embeddings)
    return store


def test_chroma_store_add_and_count(store_with_data):
    assert store_with_data.count() == 3


def test_chroma_store_query(store_with_data, embedder):
    query_vec = embedder.embed_single("disease germs")
    results = store_with_data.query(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0].score > 0


def test_dense_retriever(store_with_data, embedder):
    retriever = DenseRetriever(embedding=embedder, vectorstore=store_with_data)
    results = retriever.retrieve("disease germs", top_k=2)
    assert len(results) == 2
    # The most relevant chunk should mention germs
    assert "germ" in results[0].chunk.text.lower()
