from collections import defaultdict

import numpy as np
from langchain_core.documents import Document

from .index_store import SearchIndex
from .onnx_models import embed, rerank


def hybrid_search(index: SearchIndex, query: str, top_k: int = 10):
    all_chunks = index.chunks

    # 1. Dense Retrieval (embeddings are unit length, so dot product = cosine)
    dense_scores = index.embeddings @ embed([query])[0]
    dense_indices = np.argsort(-dense_scores)[:top_k]
    dense_results = [all_chunks[i] for i in dense_indices]

    # 2. Sparse Retrieval
    tokenized_query = query.lower().split()
    sparse_scores = index.bm25.get_scores(tokenized_query)
    sparse_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:top_k]
    sparse_results = [all_chunks[i] for i in sparse_indices]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = defaultdict(float)
    chunk_map = {}

    k_constant = 60
    for results in (dense_results, sparse_results):
        for rank, doc in enumerate(results):
            cid = doc.metadata["chunk_id"]
            rrf_scores[cid] += 1.0 / (k_constant + rank)
            chunk_map[cid] = doc

    fused_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k*2]
    fused_docs = [chunk_map[cid] for cid in fused_cids]

    # 4. Cross-Encoder Reranking
    cross_scores = rerank(query, [doc.page_content for doc in fused_docs])

    reranked = sorted(zip(fused_docs, cross_scores), key=lambda x: x[1], reverse=True)
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": round(float(score), 3)},
        )
        for doc, score in reranked[:top_k]
    ]
