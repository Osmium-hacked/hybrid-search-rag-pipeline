"""
Orchestrator — coordinates the Retriever, Generator, and Verifier agents.

Flow:
  1. RetrieverAgent finds context documents (with one internal retry).
  2. GeneratorAgent drafts an answer from those documents.
  3. VerifierAgent checks the answer's citations against the context.
  4. If verification fails, the Orchestrator feeds the verifier's feedback
     back into the GeneratorAgent and tries again, up to max_attempts.

This is the actual multi-agent piece: three independently-responsible
agents with a feedback loop between them, rather than a single straight-
line pipeline of function calls.
"""

from typing import Any, Dict, List

from rag_engine.agents.retriever_agent import RetrieverAgent
from rag_engine.agents.generator_agent import GeneratorAgent
from rag_engine.agents.verifier_agent import VerifierAgent


class RAGOrchestrator:
    def __init__(self, max_attempts: int = 2, top_k: int = 5):
        self.retriever = RetrieverAgent(top_k=top_k)
        self.generator = GeneratorAgent()
        self.verifier = VerifierAgent()
        self.max_attempts = max_attempts

    def run(self, query: str) -> Dict[str, Any]:
        docs = self.retriever.run(query)

        if not docs:
            return {
                "answer": "No relevant documents found.",
                "citations_verified": False,
                "retrieved_sources": [],
                "attempts": 0,
                "agent_trace": ["retriever: no documents found"],
            }

        trace: List[str] = [f"retriever: found {len(docs)} documents"]
        feedback = None
        answer = ""
        context_str = ""
        verified = False
        attempt = 0

        for attempt in range(1, self.max_attempts + 1):
            answer, context_str = self.generator.run(query, docs, feedback=feedback)
            trace.append(f"generator: attempt {attempt} produced an answer")

            result = self.verifier.run(query, answer, context_str)
            verified = result["verified"]

            if verified:
                trace.append(f"verifier: attempt {attempt} passed")
                break

            feedback = result["feedback"]
            trace.append(
                f"verifier: attempt {attempt} failed — {feedback or 'no reason given'}"
            )

        return {
            "answer": answer,
            "citations_verified": verified,
            "retrieved_sources": [d.metadata for d in docs],
            "attempts": attempt,
            "agent_trace": trace,
        }
