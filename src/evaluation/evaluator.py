"""Main evaluator that combines all evaluation methods."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from src.models import RAGResponse, RetrievedChunk, EvaluationScore, EvaluatedResponse


EvalMethod = Literal["ragas", "groundedness", "simple", "all"]


@dataclass
class EvaluatorConfig:
    method: EvalMethod = "simple"  # "ragas", "groundedness", "simple", or "all"
    llm_model: str = "gpt-4o-mini"


class Evaluator:
    """Evaluates RAG responses using configurable methods.

    Methods:
        - "simple": Just confidence score (fast, free)
        - "groundedness": LLM-as-judge groundedness check (uses API)
        - "ragas": Full Ragas evaluation (uses API, slower)
        - "all": All methods combined
    """

    def __init__(self, config: EvaluatorConfig | None = None):
        self.config = config or EvaluatorConfig()
        self._ragas_evaluator = None
        self._groundedness_checker = None

    @property
    def ragas_evaluator(self):
        if self._ragas_evaluator is None:
            from src.evaluation.ragas_evaluator import RagasEvaluator
            self._ragas_evaluator = RagasEvaluator(llm_model=self.config.llm_model)
        return self._ragas_evaluator

    @property
    def groundedness_checker(self):
        if self._groundedness_checker is None:
            from src.evaluation.groundedness import GroundednessChecker
            self._groundedness_checker = GroundednessChecker(model_name=self.config.llm_model)
        return self._groundedness_checker

    def evaluate(
        self,
        response: RAGResponse,
        retrieved_chunks: list[RetrievedChunk],
        ground_truth: str | None = None,
    ) -> EvaluatedResponse:
        """Evaluate a RAG response.

        Args:
            response: The RAGResponse to evaluate
            retrieved_chunks: The chunks that were retrieved (with scores)
            ground_truth: Optional expected answer

        Returns:
            EvaluatedResponse with the original response and evaluation scores
        """
        method = self.config.method

        confidence = self._calculate_confidence(retrieved_chunks)

        if method == "simple":
            scores = EvaluationScore(
                groundedness=0.0,
                relevance=0.0,
                faithfulness=0.0,
                confidence=confidence,
            )

        elif method == "groundedness":
            groundedness_score = self.groundedness_checker.get_score(
                response, retrieved_chunks
            )
            scores = EvaluationScore(
                groundedness=groundedness_score,
                relevance=0.0,
                faithfulness=groundedness_score,  
                confidence=confidence,
            )

        elif method == "ragas":
            ragas_scores = self.ragas_evaluator.evaluate_response(
                response, retrieved_chunks, ground_truth
            )
            scores = EvaluationScore(
                groundedness=ragas_scores.groundedness,
                relevance=ragas_scores.relevance,
                faithfulness=ragas_scores.faithfulness,
                confidence=confidence,
            )

        elif method == "all":
            ragas_scores = self.ragas_evaluator.evaluate_response(
                response, retrieved_chunks, ground_truth
            )
            groundedness_score = self.groundedness_checker.get_score(
                response, retrieved_chunks
            )

            scores = EvaluationScore(
                groundedness=groundedness_score,
                relevance=ragas_scores.relevance,
                faithfulness=ragas_scores.faithfulness,
                confidence=confidence,
            )

        else:
            raise ValueError(f"Unknown evaluation method: {method}")

        return EvaluatedResponse(
            response=response,
            evaluation=scores,
        )

    def _calculate_confidence(self, retrieved_chunks: list[RetrievedChunk]) -> float:
        """Calculate confidence as average retrieval similarity score."""
        if not retrieved_chunks:
            return 0.0
        return sum(rc.score for rc in retrieved_chunks) / len(retrieved_chunks)

    def evaluate_batch(
        self,
        responses: list[tuple[RAGResponse, list[RetrievedChunk]]],
        ground_truths: list[str | None] | None = None,
    ) -> list[EvaluatedResponse]:
        """Evaluate multiple responses.

        Args:
            responses: List of (RAGResponse, retrieved_chunks) tuples
            ground_truths: Optional list of expected answers

        Returns:
            List of EvaluatedResponse objects
        """
        if ground_truths is None:
            ground_truths = [None] * len(responses)

        results = []
        for (response, chunks), gt in zip(responses, ground_truths):
            result = self.evaluate(response, chunks, gt)
            results.append(result)

        return results
