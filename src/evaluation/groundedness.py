"""Groundedness checker using LLM-as-judge pattern."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI

from src.models import RAGResponse, RetrievedChunk


GROUNDEDNESS_SYSTEM_PROMPT = """You are an expert fact-checker. Your job is to verify if claims in an answer are supported by the provided source chunks.

For each claim in the answer, determine if it is:
- SUPPORTED: The claim is directly stated or can be reasonably inferred from the source chunks
- NOT_SUPPORTED: The claim is not found in or contradicts the source chunks
- PARTIAL: The claim is partially supported but missing key details

Be strict. Only mark as SUPPORTED if the source chunks clearly support the claim."""

GROUNDEDNESS_USER_PROMPT = """Evaluate the groundedness of this answer based on the source chunks.

QUESTION: {question}

ANSWER: {answer}

SOURCE CHUNKS:
{chunks}

Respond in JSON format:
{{
    "claims": [
        {{"claim": "...", "verdict": "SUPPORTED|NOT_SUPPORTED|PARTIAL", "evidence": "quote or explanation"}}
    ],
    "overall_score": 0.0 to 1.0,
    "explanation": "brief overall assessment"
}}"""


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str  # SUPPORTED, NOT_SUPPORTED, PARTIAL
    evidence: str


@dataclass
class GroundednessResult:
    claims: list[ClaimVerdict]
    overall_score: float
    explanation: str


class GroundednessChecker:
    """Checks if an answer is grounded in the retrieved context using LLM-as-judge."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def check(
        self,
        response: RAGResponse,
        retrieved_chunks: list[RetrievedChunk],
    ) -> GroundednessResult:
        """Check groundedness of a RAG response.

        Args:
            response: The RAGResponse with question and answer
            retrieved_chunks: The chunks used to generate the answer

        Returns:
            GroundednessResult with claim-level and overall assessment
        """
        chunks_text = self._format_chunks(retrieved_chunks)

        user_prompt = GROUNDEDNESS_USER_PROMPT.format(
            question=response.question,
            answer=response.answer,
            chunks=chunks_text,
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": GROUNDEDNESS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            result_json = json.loads(completion.choices[0].message.content)

            claims = [
                ClaimVerdict(
                    claim=c.get("claim", ""),
                    verdict=c.get("verdict", "NOT_SUPPORTED"),
                    evidence=c.get("evidence", ""),
                )
                for c in result_json.get("claims", [])
            ]

            return GroundednessResult(
                claims=claims,
                overall_score=float(result_json.get("overall_score", 0.0)),
                explanation=result_json.get("explanation", ""),
            )

        except Exception as e:
            print(f"Groundedness check failed: {e}")
            return GroundednessResult(
                claims=[],
                overall_score=0.0,
                explanation=f"Evaluation failed: {str(e)}",
            )

    def _format_chunks(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks for inclusion in the prompt."""
        parts = []
        for i, rc in enumerate(chunks, 1):
            parts.append(
                f"[Chunk {i} | Page {rc.chunk.page_number}]\n{rc.chunk.text[:1000]}"
            )
        return "\n\n".join(parts)

    def get_score(
        self,
        response: RAGResponse,
        retrieved_chunks: list[RetrievedChunk],
    ) -> float:
        """Get just the overall groundedness score (0.0 to 1.0)."""
        result = self.check(response, retrieved_chunks)
        return result.overall_score
