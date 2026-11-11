"""
GeneratorAgent — wraps the existing generate_answer function. Accepts
optional feedback from the VerifierAgent so the Orchestrator can loop:
generate -> verify -> (if failed) regenerate with feedback -> verify again.
"""

from typing import List, Optional, Tuple

from rag_engine.generation import generate_answer


class GeneratorAgent:
    def run(
        self,
        query: str,
        docs: List,
        feedback: Optional[str] = None,
    ) -> Tuple[str, str]:
        if feedback:
            # Fold the verifier's feedback into the query so the next
            # generation attempt corrects for what failed verification.
            query = (
                f"{query}\n\n"
                f"Note: a previous answer to this question failed a citation "
                f"check for this reason: {feedback}\n"
                f"Produce a corrected answer that only makes claims directly "
                f"supported by the provided context."
            )

        answer, context_str = generate_answer(query, docs)
        return answer, context_str
