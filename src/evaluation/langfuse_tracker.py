"""Langfuse integration for tracing and observability."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.models import RAGResponse, EvaluationScore, EvaluatedResponse


class LangfuseTracker:
    """Tracks RAG pipeline execution and evaluation scores in Langfuse."""

    def __init__(self):
        self.enabled = self._check_enabled()
        self._client = None

    def _check_enabled(self) -> bool:
        """Check if Langfuse credentials are configured."""
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        return bool(public_key and secret_key)

    @property
    def client(self):
        """Lazy-init Langfuse client."""
        if self._client is None and self.enabled:
            from langfuse import Langfuse
            self._client = Langfuse()
        return self._client

    def log_full_evaluation(
        self,
        ocr_model: str,
        chunking_strategy: str,
        evaluated_responses: list[EvaluatedResponse],
    ) -> None:
        """Log a complete evaluation run with all questions and scores."""
        if not self.enabled:
            print("Langfuse not configured — skipping trace logging")
            return

        try:
            trace_name = f"evaluation_{ocr_model}__{chunking_strategy}"
            with self.client.start_as_current_span(
                name=trace_name,
                input={
                    "ocr_model": ocr_model,
                    "chunking_strategy": chunking_strategy,
                    "num_questions": len(evaluated_responses),
                },
            ) as span:
                for i, er in enumerate(evaluated_responses):
                    with self.client.start_as_current_span(
                        name=f"question_{i+1}",
                        input={"question": er.response.question},
                    ) as q_span:
                        q_span.update(
                            output={"answer": er.response.answer[:500]},
                            metadata={
                                "supporting_chunks": [
                                    sc.chunk_id for sc in er.response.supporting_chunks
                                ],
                            },
                        )

                # Log aggregate scores
                if evaluated_responses:
                    avg_faith = sum(er.evaluation.faithfulness for er in evaluated_responses) / len(evaluated_responses)
                    avg_rel = sum(er.evaluation.relevance for er in evaluated_responses) / len(evaluated_responses)
                    avg_ground = sum(er.evaluation.groundedness for er in evaluated_responses) / len(evaluated_responses)
                    conf_vals = [er.evaluation.confidence for er in evaluated_responses if er.evaluation.confidence]
                    avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0

                    span.update(
                        output={
                            "avg_faithfulness": avg_faith,
                            "avg_relevance": avg_rel,
                            "avg_groundedness": avg_ground,
                            "avg_confidence": avg_conf,
                        }
                    )

                    # Log scores
                    self.client.create_score(
                        trace_id=self.client.get_current_trace_id(),
                        name="avg_faithfulness",
                        value=avg_faith,
                    )
                    self.client.create_score(
                        trace_id=self.client.get_current_trace_id(),
                        name="avg_relevance",
                        value=avg_rel,
                    )
                    self.client.create_score(
                        trace_id=self.client.get_current_trace_id(),
                        name="avg_groundedness",
                        value=avg_ground,
                    )
                    self.client.create_score(
                        trace_id=self.client.get_current_trace_id(),
                        name="avg_confidence",
                        value=avg_conf,
                    )

            # Flush to ensure data is sent
            self.client.flush()

        except Exception as e:
            print(f"Langfuse logging failed: {e}")


_tracker: LangfuseTracker | None = None


def get_tracker() -> LangfuseTracker:
    """Get or create the global Langfuse tracker."""
    global _tracker
    if _tracker is None:
        _tracker = LangfuseTracker()
    return _tracker
