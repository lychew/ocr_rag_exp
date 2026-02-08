"""Tests for generation components (unit tests that don't call OpenAI)."""

from src.generation.prompt_templates import format_context, RAG_USER_PROMPT
from src.models import TextChunk, RetrievedChunk


def test_format_context():
    chunks = [
        RetrievedChunk(
            chunk=TextChunk(chunk_id="c1", text="First chunk text.", page_number=1),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=TextChunk(chunk_id="c2", text="Second chunk text.", page_number=2),
            score=0.8,
        ),
    ]
    ctx = format_context(chunks)
    assert "c1" in ctx
    assert "c2" in ctx
    assert "First chunk text." in ctx
    assert "Page: 1" in ctx


def test_user_prompt_template():
    prompt = RAG_USER_PROMPT.format(question="test?", context="some context")
    assert "test?" in prompt
    assert "some context" in prompt
