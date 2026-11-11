"""
RetrieverAgent — wraps the existing hybrid_search retrieval function with a
simple agentic retry: if the first search comes back thin (too few
results), it reformulates the query once and searches again before giving
up. This is what turns a plain retrieval call into a small decision-making
agent rather than a straight-line function call.
"""

from typing import List

from rag_engine.index_store import SearchIndex
from rag_engine.retrieval import hybrid_search


class RetrieverAgent:
    def __init__(self, min_results: int = 2, top_k: int = 5):
        self.min_results = min_results
        self.top_k = top_k

    def run(self, query: str, index: SearchIndex) -> List:
        docs = hybrid_search(index, query, top_k=self.top_k)

        if len(docs) >= self.min_results:
            return docs

        # Weak retrieval — reformulate once and retry.
        reformulated = self._reformulate_query(query)
        retry_docs = hybrid_search(index, reformulated, top_k=self.top_k)

        # Keep whichever attempt returned more (or any) results.
        return retry_docs if len(retry_docs) > len(docs) else docs

    def _reformulate_query(self, query: str) -> str:
        """
        Cheap, deterministic reformulation: strips filler words so the
        search leans on the content-bearing terms. Swap this out for an
        LLM call (e.g. "rewrite this query to surface more relevant
        documents") for a stronger agent — the interface (str in, str out)
        stays the same either way.
        """
        stopwords = {"the", "a", "an", "please", "can", "you", "tell", "me", "about"}
        tokens = [t for t in query.split() if t.lower() not in stopwords]
        return " ".join(tokens) if tokens else query
