"""
VerifierAgent — wraps the judge model's verify_citations call and normalizes
its verdict into a structured pass/fail + feedback result the Orchestrator
can act on. It fails closed: if the judge can't be reached or returns
something unusable, the answer is reported as not verified.
"""

import logging
from typing import List, Literal, TypedDict

from rag_engine.generation import verify_citations

logger = logging.getLogger(__name__)


class VerificationResult(TypedDict):
    verified: bool
    verdict: Literal["supported", "unsupported", "unavailable"]
    reasoning: str
    unsupported_claims: List[str]
    feedback: str


class VerifierAgent:
    def run(self, query: str, answer: str, context_str: str) -> VerificationResult:
        try:
            result = verify_citations(query, answer, context_str)
        except Exception:
            logger.exception("Judge call failed")
            return {
                "verified": False,
                "verdict": "unavailable",
                "reasoning": "The judge model could not be reached, so this answer was not checked.",
                "unsupported_claims": [],
                "feedback": "",
            }

        # A verdict that contradicts its own list of bad claims doesn't pass.
        verified = result.supported and not result.unsupported_claims
        feedback = ""
        if not verified:
            feedback = " ".join([result.reasoning, *result.unsupported_claims])

        return {
            "verified": verified,
            "verdict": "supported" if verified else "unsupported",
            "reasoning": result.reasoning,
            "unsupported_claims": result.unsupported_claims,
            "feedback": feedback,
        }
