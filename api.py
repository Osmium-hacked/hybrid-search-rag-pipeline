from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine.ingestion import ingest_documents
from rag_engine.orchestrator import RAGOrchestrator

app = FastAPI(title="Production RAG API")
orchestrator = RAGOrchestrator()


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
    result = orchestrator.run(request.query)
    return result
