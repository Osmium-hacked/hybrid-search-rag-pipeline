from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine.ingestion import ingest_documents
from rag_engine.retrieval import hybrid_search
from rag_engine.generation import generate_answer, verify_citations

app = FastAPI(title="Production RAG API")

class AskRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    file_paths: list[str]

@app.post("/v1/ingest")
async def ingest(request: IngestRequest):
    try:
        chunk_count = ingest_documents(request.file_paths)
        return {"message": "Success", "chunks_indexed": chunk_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ask")
async def ask(request: AskRequest):
    docs = hybrid_search(request.query, top_k=5)
    if not docs:
        return {"answer": "No relevant documents found."}
        
    answer, context_str = generate_answer(request.query, docs)
    verification = verify_citations(request.query, answer, context_str)
    
    return {
        "answer": answer,
        "citations_verified": "VERIFIED" in verification,
        "retrieved_sources": [d.metadata for d in docs]
    }