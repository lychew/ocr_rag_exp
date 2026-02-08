"""Tests for embedding provider."""

import pytest

from src.embedding.sentence_transformer import SentenceTransformerEmbedding


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedding(model_name="BAAI/bge-small-en-v1.5")


def test_embed_returns_list(embedder):
    vecs = embedder.embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) > 0  # embedding dimension


def test_embed_multiple(embedder):
    vecs = embedder.embed(["first", "second", "third"])
    assert len(vecs) == 3


def test_embed_single(embedder):
    vec = embedder.embed_single("test query")
    assert isinstance(vec, list)
    assert len(vec) > 0
