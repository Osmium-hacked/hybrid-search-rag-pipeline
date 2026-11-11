import logging
from pathlib import Path

import groq
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_engine.config import (
    JUDGE_MODEL,
    LLM_MODEL,
    MAX_DOCUMENTS,
    MAX_FILE_CHUNKS,
    MAX_UPLOAD_BYTES,
    MAX_WORKSPACE_CHUNKS,
    SAMPLE_NAME,
    TRANSCRIPTION_MODEL,
    UPLOADS_ENABLED,
    VISION_MODEL,
    WORKSPACE_TTL_DAYS,
)
from rag_engine.extractors import FORMAT_GROUPS, READERS, ExtractionError, UnsupportedFormat
from rag_engine.index_store import WORKSPACE_ID, delete_workspace, load_workspace
from rag_engine.ingestion import LimitExceeded, add_files, add_sample, describe, remove_document
from rag_engine.orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="HyRAG API")
orchestrator = RAGOrchestrator()


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def workspace_id(x_workspace_id: str = Header(default="")) -> str:
    """Each browser sends the random ID of its own private workspace."""
    if not WORKSPACE_ID.fullmatch(x_workspace_id):
        raise HTTPException(status_code=400, detail="This request is missing a valid workspace ID.")
    return x_workspace_id


def writable_workspace(workspace: str = Depends(workspace_id)) -> str:
    if not UPLOADS_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Uploads need a storage connection that this deployment doesn't have yet.",
        )
    return workspace


def _provider_errors(error: Exception, action: str) -> HTTPException:
    if isinstance(error, groq.RateLimitError):
        return HTTPException(status_code=503, detail="The model provider is rate limiting requests. Try again in a minute.")
    if isinstance(error, (groq.InternalServerError, groq.APIConnectionError)):
        logger.exception("Groq was unavailable")
        return HTTPException(status_code=503, detail="The model provider is busy right now. Try again in a minute.")
    if isinstance(error, groq.AuthenticationError):
        logger.exception("Groq rejected the API key")
        return HTTPException(status_code=502, detail="The model provider rejected the configured API key.")
    logger.exception("%s failed", action)
    return HTTPException(status_code=500, detail=f"Something went wrong while {action}.")


def _workspace_body(workspace: str) -> dict:
    index = load_workspace(workspace)
    return {
        "documents": describe(index),
        "total_chunks": len(index.chunks) if index else 0,
    }


@app.get("/api/v1/status")
def status():
    return {
        "models": {
            "generator": LLM_MODEL,
            "judge": JUDGE_MODEL,
            "vision": VISION_MODEL,
            "transcription": TRANSCRIPTION_MODEL,
        },
        "uploads": {
            "enabled": UPLOADS_ENABLED,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_file_chunks": MAX_FILE_CHUNKS,
            "max_workspace_chunks": MAX_WORKSPACE_CHUNKS,
            "max_documents": MAX_DOCUMENTS,
            "retention_days": WORKSPACE_TTL_DAYS,
            "extensions": sorted(READERS),
            "formats": FORMAT_GROUPS,
        },
        "sample": SAMPLE_NAME,
    }


@app.get("/api/v1/workspace")
def get_workspace(workspace: str = Depends(workspace_id)):
    return _workspace_body(workspace)


@app.delete("/api/v1/workspace")
def clear_workspace(workspace: str = Depends(writable_workspace)):
    delete_workspace(workspace)
    return _workspace_body(workspace)


@app.post("/api/v1/workspace/documents")
def upload_documents(
    files: list[UploadFile] = File(...),
    workspace: str = Depends(writable_workspace),
):
    payload = []
    for upload in files:
        name = Path(upload.filename or "").name.strip()[:200]
        if not name:
            raise HTTPException(status_code=400, detail="Every file needs a name.")
        data = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} is larger than the upload limit.")
        if not data:
            raise HTTPException(status_code=422, detail=f"{name} is empty.")
        payload.append((name, data))

    try:
        added = add_files(workspace, payload)
    except UnsupportedFormat as e:
        raise HTTPException(status_code=415, detail=str(e))
    except (ExtractionError, LimitExceeded) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _provider_errors(e, "reading that file")
    return {"added": added, **_workspace_body(workspace)}


@app.post("/api/v1/workspace/sample")
def load_sample(workspace: str = Depends(writable_workspace)):
    try:
        added = add_sample(workspace)
    except LimitExceeded as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"added": [added], **_workspace_body(workspace)}


@app.delete("/api/v1/workspace/documents")
def delete_document(name: str, workspace: str = Depends(writable_workspace)):
    if not remove_document(workspace, name):
        raise HTTPException(status_code=404, detail="That document isn't in this workspace.")
    return _workspace_body(workspace)


@app.post("/api/v1/ask")
def ask(request: AskRequest, workspace: str = Depends(workspace_id)):
    index = load_workspace(workspace)
    if index is None:
        raise HTTPException(status_code=409, detail="Add a document before asking a question.")
    try:
        return orchestrator.run(request.query, index)
    except Exception as e:
        raise _provider_errors(e, "answering that question")
