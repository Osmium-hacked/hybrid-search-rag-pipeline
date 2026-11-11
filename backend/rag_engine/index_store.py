"""
Loads and saves search indexes. Each visitor's workspace is one .npz file
holding its chunks and their embeddings, so a save replaces it atomically.

On Vercel the filesystem is read-only, so workspaces live in a private
Vercel Blob store. Locally they're files under backend/index/workspaces.
The sample document's index ships with the deployment and is read-only.
"""

import io
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import (
    BLOB_ENABLED,
    BLOB_WORKSPACE_PREFIX,
    INDEX_REFRESH_SECONDS,
    LOCAL_WORKSPACE_DIR,
    SAMPLE_INDEX_PATH,
    WORKSPACE_TTL_DAYS,
)

logger = logging.getLogger(__name__)

# Workspace IDs are random, unguessable tokens made by the browser. Checking
# the shape also keeps them safe to use in file and blob paths.
WORKSPACE_ID = re.compile(r"[A-Za-z0-9_-]{16,64}")

CACHE_SIZE = 32
PRUNE_INTERVAL_SECONDS = 3600


@dataclass(frozen=True)
class SearchIndex:
    chunks: list[Document]
    embeddings: np.ndarray
    bm25: BM25Okapi
    version: Optional[str] = None


def build_index(chunks: list[Document], embeddings: np.ndarray, version: Optional[str] = None) -> SearchIndex:
    bm25 = BM25Okapi([c.page_content.lower().split() for c in chunks])
    return SearchIndex(chunks=chunks, embeddings=embeddings, bm25=bm25, version=version)


def _serialize(chunks: list[Document], embeddings: np.ndarray) -> bytes:
    records = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, chunks=np.array(json.dumps(records)), embeddings=embeddings)
    return buffer.getvalue()


def _deserialize(data: bytes, version: Optional[str] = None) -> SearchIndex:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        records = json.loads(str(npz["chunks"]))
        embeddings = npz["embeddings"]
    chunks = [Document(page_content=r["text"], metadata=r["metadata"]) for r in records]
    return build_index(chunks, embeddings, version)


@lru_cache(maxsize=1)
def sample_index() -> SearchIndex:
    return _deserialize(SAMPLE_INDEX_PATH.read_bytes())


def _blob_path(workspace: str) -> str:
    return f"{BLOB_WORKSPACE_PREFIX}{workspace}/index.npz"


def _local_path(workspace: str):
    return LOCAL_WORKSPACE_DIR / f"{workspace}.npz"


_lock = threading.Lock()
# workspace -> (index or None when empty, monotonic time it was checked)
_cache: "OrderedDict[str, tuple[Optional[SearchIndex], float]]" = OrderedDict()
_pruned_at = float("-inf")


def _remember(workspace: str, index: Optional[SearchIndex]) -> None:
    with _lock:
        _cache[workspace] = (index, time.monotonic())
        _cache.move_to_end(workspace)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def _read(workspace: str, current: Optional[SearchIndex]) -> Optional[SearchIndex]:
    if BLOB_ENABLED:
        from vercel.blob.sync import BlobNotFoundError, get_blob, head_blob

        path = _blob_path(workspace)
        try:
            version = head_blob(path).etag
            if current is not None and current.version == version:
                return current
            return _deserialize(get_blob(path, access="private").body, version)
        except BlobNotFoundError:
            return None

    path = _local_path(workspace)
    if not path.exists():
        return None
    version = str(path.stat().st_mtime_ns)
    if current is not None and current.version == version:
        return current
    return _deserialize(path.read_bytes(), version)


def load_workspace(workspace: str, fresh: bool = False) -> Optional[SearchIndex]:
    """Returns the workspace's index, or None if it has no documents.

    A cached copy is reused for INDEX_REFRESH_SECONDS. Pass fresh=True before
    changing a workspace, so a save never builds on a stale copy.
    """
    with _lock:
        entry = _cache.get(workspace)
    if entry and not fresh and time.monotonic() - entry[1] < INDEX_REFRESH_SECONDS:
        return entry[0]
    index = _read(workspace, entry[0] if entry else None)
    _remember(workspace, index)
    return index


def save_workspace(workspace: str, chunks: list[Document], embeddings: np.ndarray) -> None:
    if not chunks:
        delete_workspace(workspace)
        return

    data = _serialize(chunks, embeddings)
    if BLOB_ENABLED:
        from vercel.blob.sync import put_blob

        version = put_blob(
            _blob_path(workspace),
            data,
            access="private",
            add_random_suffix=False,
            allow_overwrite=True,
            content_type="application/octet-stream",
        ).etag
    else:
        path = _local_path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        version = str(path.stat().st_mtime_ns)
    _remember(workspace, build_index(chunks, embeddings, version))


def delete_workspace(workspace: str) -> None:
    if BLOB_ENABLED:
        from vercel.blob.sync import BlobNotFoundError, delete_blob

        try:
            delete_blob(_blob_path(workspace))
        except BlobNotFoundError:
            pass
    else:
        _local_path(workspace).unlink(missing_ok=True)
    _remember(workspace, None)


def prune_expired_workspaces() -> None:
    """Deletes workspaces that haven't changed for WORKSPACE_TTL_DAYS.

    Runs at most once an hour per instance, and never fails the caller.
    """
    global _pruned_at
    with _lock:
        if time.monotonic() - _pruned_at < PRUNE_INTERVAL_SECONDS:
            return
        _pruned_at = time.monotonic()

    cutoff = datetime.now(timezone.utc) - timedelta(days=WORKSPACE_TTL_DAYS)
    expired: list = []
    try:
        if BLOB_ENABLED:
            from vercel.blob.sync import delete_blob, query_blobs

            expired = [
                blob.pathname
                for blob in query_blobs(prefix=BLOB_WORKSPACE_PREFIX, page_size=1000)
                if blob.uploaded_at.astimezone(timezone.utc) < cutoff
            ]
            if expired:
                delete_blob(expired)
        elif LOCAL_WORKSPACE_DIR.exists():
            expired = [
                path
                for path in LOCAL_WORKSPACE_DIR.glob("*.npz")
                if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < cutoff
            ]
            for path in expired:
                path.unlink(missing_ok=True)
        if expired:
            logger.info("Deleted %d expired workspaces", len(expired))
    except Exception:
        logger.exception("Couldn't prune expired workspaces")
