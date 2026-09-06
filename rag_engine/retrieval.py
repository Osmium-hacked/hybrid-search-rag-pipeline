import pickle
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from .config import CHROMA_PERSIST_DIR, BM25_INDEX_PATH, EMBEDDING_MODEL, RERANKER_MODEL

# Local cross-encoder reranker
reranker = CrossEncoder(RERANKER_MODEL)
# Local embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def hybrid_search(query: str, top_k: int = 10):
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
        bm25, all_chunks = data["bm25"], data["chunks"]

    # 1. Dense Retrieval
    dense_results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    # 2. Sparse Retrieval
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:top_k]
    sparse_results = [all_chunks[i] for i in sparse_indices]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = defaultdict(float)
    chunk_map = {}
    
    k_constant = 60
    for rank, (doc, _) in enumerate(dense_results):
        cid = doc.metadata["chunk_id"]
        rrf_scores[cid] += 1.0 / (k_constant + rank)
        chunk_map[cid] = doc
        
    for rank, doc in enumerate(sparse_results):
        cid = doc.metadata["chunk_id"]
        rrf_scores[cid] += 1.0 / (k_constant + rank)
        chunk_map[cid] = doc

    fused_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k*2]
    fused_docs = [chunk_map[cid] for cid in fused_cids]

    # 4. Cross-Encoder Reranking
    pairs = [[query, doc.page_content] for doc in fused_docs]
    cross_scores = reranker.predict(pairs)
    
    reranked = sorted(zip(fused_docs, cross_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_k]]