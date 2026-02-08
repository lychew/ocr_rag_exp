"""Prompt templates for RAG generation."""

from enum import Enum


class PromptStrategy(str, Enum):
    BASIC = "basic"
    COT = "cot"  # Chain-of-Thought
    FEW_SHOT = "few_shot"


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_BASIC = """You are a helpful assistant that answers questions about the book "Principles of Public Health".
You MUST ground every claim in the provided context chunks. Do not use any outside knowledge.
If the context does not contain enough information, say so explicitly."""

SYSTEM_PROMPT_COT = """You are a helpful assistant that answers questions about the book "Principles of Public Health".
You MUST ground every claim in the provided context chunks. Do not use any outside knowledge.
Think step-by-step before providing your final answer.
If the context does not contain enough information, say so explicitly."""

SYSTEM_PROMPT_FEW_SHOT = SYSTEM_PROMPT_BASIC  # Same system prompt, examples in user prompt


# =============================================================================
# USER PROMPTS
# =============================================================================

USER_PROMPT_BASIC = """Answer the following question using ONLY the provided context chunks.

Question: {question}

Context chunks:
{context}

Instructions:
- Answer clearly and concisely based only on the context above.
- Reference specific chunks that support your answer.
- If the context is insufficient, say "Based on the available context, I cannot fully answer this question."
"""

USER_PROMPT_COT = """Answer the following question using ONLY the provided context chunks.

Question: {question}

Context chunks:
{context}

Instructions:
Think through this step-by-step:

1. **Identify Relevant Chunks**: Which chunks contain information related to the question?
2. **Extract Key Facts**: What are the main points from those chunks?
3. **Synthesize Answer**: Combine the facts to answer the question.

Provide your reasoning, then give your final answer.
"""

USER_PROMPT_FEW_SHOT = """Answer the following question using ONLY the provided context chunks.

Here is an example of how to answer:

---
Example Question: What causes tuberculosis?

Example Context:
[Chunk 1 | ID: example_001 | Page: 15]
Tuberculosis is caused by a specific germ, the tubercle bacillus, which enters the body through the air we breathe.

Example Answer: According to the book, tuberculosis is caused by the tubercle bacillus, a specific germ that enters the body through the air we breathe (Chunk 1, Page 15).
---

Now answer the following question:

Question: {question}

Context chunks:
{context}

Answer based only on the context above, citing the relevant chunks.
"""


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

PROMPT_STRATEGIES = {
    PromptStrategy.BASIC: {
        "system": SYSTEM_PROMPT_BASIC,
        "user": USER_PROMPT_BASIC,
    },
    PromptStrategy.COT: {
        "system": SYSTEM_PROMPT_COT,
        "user": USER_PROMPT_COT,
    },
    PromptStrategy.FEW_SHOT: {
        "system": SYSTEM_PROMPT_FEW_SHOT,
        "user": USER_PROMPT_FEW_SHOT,
    },
}


def get_prompts(strategy: str | PromptStrategy = PromptStrategy.BASIC) -> dict:
    """Get system and user prompts for a given strategy."""
    if isinstance(strategy, str):
        strategy = PromptStrategy(strategy)
    return PROMPT_STRATEGIES[strategy]


def format_context(chunks) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts = []
    for i, rc in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | ID: {rc.chunk.chunk_id} | Page: {rc.chunk.page_number}]\n{rc.chunk.text}"
        )
    return "\n\n".join(parts)


# Keep backwards compatibility
RAG_SYSTEM_PROMPT = SYSTEM_PROMPT_BASIC
RAG_USER_PROMPT = USER_PROMPT_BASIC
