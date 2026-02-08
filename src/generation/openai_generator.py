"""OpenAI LLM generator (GPT-4o-mini default)."""

from __future__ import annotations

import os

from openai import OpenAI

from src.generation.base import BaseGenerator
from src.generation.prompt_templates import (
    PromptStrategy,
    format_context,
    get_prompts,
)
from src.models import RAGResponse, RetrievedChunk, SupportingChunk


class OpenAIGenerator(BaseGenerator):

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        prompt_strategy: str = "basic",
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_strategy = prompt_strategy
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> RAGResponse:
        # Get prompts for the selected strategy
        prompts = get_prompts(self.prompt_strategy)
        system_prompt = prompts["system"]
        user_template = prompts["user"]

        context = format_context(chunks)
        user_msg = user_template.format(question=question, context=context)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer_text = response.choices[0].message.content.strip()

        supporting = [
            SupportingChunk(
                chunk_id=rc.chunk.chunk_id,
                page=rc.chunk.page_number,
            )
            for rc in chunks
        ]

        return RAGResponse(
            question=question,
            answer=answer_text,
            supporting_chunks=supporting,
        )
