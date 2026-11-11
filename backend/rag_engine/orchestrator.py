"""
Orchestrator — coordinates the Retriever, Generator, and Verifier agents.

Flow:
  1. RetrieverAgent finds context documents (with one internal retry).
  2. GeneratorAgent drafts an answer from those documents.
  3. VerifierAgent has a separate judge model check the answer against the
     context.
  4. If verification fails, the Orchestrator feeds the judge's feedback
     back into the GeneratorAgent and tries again, up to max_attempts.

This is the actual multi-agent piece: three independently-responsible
agents with a feedback loop between them, rather than a single straight-
line pipeline of function calls.
"""

import time
from typing import Any, Dict, List

from rag_engine.agents.retriever_agent import RetrieverAgent
from rag_engine.agents.generator_agent import GeneratorAgent
from rag_engine.agents.verifier_agent import VerifierAgent
from rag_engine.config import JUDGE_MODEL, LLM_MODEL
from rag_engine.index_store import SearchIndex


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _trace_step(agent: str, status: str, detail: str, started: float, attempt: int = 1) -> Dict[str, Any]:
    return {
        "agent": agent,
        "attempt": attempt,
        "status": status,
        "detail": detail,
        "duration_ms": _elapsed_ms(started),
    }


class RAGOrchestrator:
    def __init__(self, max_attempts: int = 2, top_k: int = 5):
        self.retriever = RetrieverAgent(top_k=top_k)
        self.generator = GeneratorAgent()
        self.verifier = VerifierAgent()
        self.max_attempts = max_attempts

    def run(self, query: str, index: SearchIndex) -> Dict[str, Any]:
        run_started = time.perf_counter()
        step_started = run_started
        docs = self.retriever.run(query, index)

        models = {"generator": LLM_MODEL, "judge": JUDGE_MODEL}

        if not docs:
            return {
                "answer": "No relevant documents found.",
                "citations_verified": False,
                "judge": {
                    "verdict": "unavailable",
                    "reasoning": "There was nothing to check because no documents were found.",
                    "unsupported_claims": [],
                },
                "models": models,
                "retrieved_sources": [],
                "attempts": 0,
                "agent_trace": [_trace_step("retriever", "failed", "No documents found", step_started)],
                "duration_ms": _elapsed_ms(run_started),
            }

        trace: List[Dict[str, Any]] = [
            _trace_step("retriever", "passed", f"Found {len(docs)} passages", step_started)
        ]
        feedback = None
        answer = ""
        result = None
        attempt = 0

        for attempt in range(1, self.max_attempts + 1):
            step_started = time.perf_counter()
            answer, context_str = self.generator.run(query, docs, feedback=feedback)
            trace.append(_trace_step("generator", "passed", "Drafted an answer", step_started, attempt))

            step_started = time.perf_counter()
            result = self.verifier.run(query, answer, context_str)

            if result["verified"]:
                trace.append(_trace_step("judge", "passed", "Every claim is supported", step_started, attempt))
                break

            trace.append(_trace_step("judge", "failed", result["reasoning"], step_started, attempt))

            # Regenerating won't help if the judge itself is down.
            if result["verdict"] == "unavailable":
                break

            feedback = result["feedback"]

        return {
            "answer": answer,
            "citations_verified": result["verified"],
            "judge": {
                "verdict": result["verdict"],
                "reasoning": result["reasoning"],
                "unsupported_claims": result["unsupported_claims"],
            },
            "models": models,
            "retrieved_sources": [{**d.metadata, "content": d.page_content} for d in docs],
            "attempts": attempt,
            "agent_trace": trace,
            "duration_ms": _elapsed_ms(run_started),
        }
