"""Ragas evaluation metrics for RAG pipeline quality."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.models import RAGResponse, RetrievedChunk, EvaluationScore


@dataclass
class RagasResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float


class RagasEvaluator:
    """Evaluates RAG responses using Ragas metrics."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm_model = llm_model
        self._llm = None
        self._embeddings = None

    @property
    def llm(self):
        """Lazy-load LLM wrapper for Ragas."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            chat = ChatOpenAI(model=self.llm_model, temperature=0)
            self._llm = LangchainLLMWrapper(chat)
        return self._llm

    @property
    def embeddings(self):
        """Lazy-load embeddings wrapper for Ragas."""
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings(model="text-embedding-3-small")
            self._embeddings = LangchainEmbeddingsWrapper(emb)
        return self._embeddings

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RagasResult:
        """Evaluate a single question-answer pair.

        Args:
            question: The input question
            answer: The generated answer
            contexts: List of retrieved chunk texts
            ground_truth: Optional ground truth answer for reference

        Returns:
            RagasResult with faithfulness, answer_relevancy, context_precision
        """
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth or answer,
        )

        dataset = EvaluationDataset(samples=[sample])

        metrics = [faithfulness, answer_relevancy, context_precision]

        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            scores = results.to_pandas().iloc[0]

            return RagasResult(
                faithfulness=float(scores.get("faithfulness", 0.0)),
                answer_relevancy=float(scores.get("answer_relevancy", 0.0)),
                context_precision=float(scores.get("context_precision", 0.0)),
            )
        except Exception as e:
            print(f"Ragas evaluation failed: {e}")
            return RagasResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
            )

    def evaluate_response(
        self,
        response: RAGResponse,
        retrieved_chunks: list[RetrievedChunk],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Evaluate a RAGResponse object.

        Args:
            response: The RAGResponse with question, answer, supporting_chunks
            retrieved_chunks: The actual retrieved chunks with text
            ground_truth: Optional expected answer

        Returns:
            EvaluationScore with all metrics
        """
        contexts = [rc.chunk.text for rc in retrieved_chunks]

        ragas_result = self.evaluate_single(
            question=response.question,
            answer=response.answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        confidence = None
        if retrieved_chunks:
            confidence = sum(rc.score for rc in retrieved_chunks) / len(retrieved_chunks)

        return EvaluationScore(
            faithfulness=ragas_result.faithfulness,
            relevance=ragas_result.answer_relevancy,
            groundedness=ragas_result.faithfulness, 
            confidence=confidence,
        )
