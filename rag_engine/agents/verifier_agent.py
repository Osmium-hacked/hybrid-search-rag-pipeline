"""
VerifierAgent — wraps the existing verify_citations function and normalizes
its output into a structured pass/fail + feedback result the Orchestrator
can act on.
"""

from typing import TypedDict

from rag_engine.generation import verify_citations


class VerificationResult(TypedDict):
    verified: bool
    feedback: str


class VerifierAgent:
    def run(self, query: str, answer: str, context_str: str) -> VerificationResult:
        raw_result = verify_citations(query, answer, context_str)

        verified = "VERIFIED" in raw_result
        feedback = "" if verified else raw_result

        return {"verified": verified, "feedback": feedback}
