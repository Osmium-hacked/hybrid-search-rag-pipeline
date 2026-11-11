"""
Chunks and embeds documents, then adds them to a workspace's index.
Re-uploading a file with the same name replaces its earlier chunks.

Rebuild the bundled sample index from the command line with:
    python -m rag_engine.ingestion ../docs/attentionisalluneed.pdf
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import MAX_DOCUMENTS, MAX_FILE_CHUNKS, MAX_WORKSPACE_CHUNKS, SAMPLE_INDEX_PATH
from .extractors import ExtractionError, extract
from .index_store import (
    SearchIndex,
    _serialize,
    load_workspace,
    prune_expired_workspaces,
    sample_index,
    save_workspace,
)
from .onnx_models import embed

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""],
)


class LimitExceeded(ValueError):
    pass


def _chunk_file(name: str, data: bytes) -> tuple[list[Document], dict]:
    extracted = extract(name, data)
    pages = []
    for section in extracted.sections:
        metadata = {"source": name, "page": section.page, "location": section.location}
        if extracted.method:
            metadata["extraction"] = extracted.method
        pages.append(Document(page_content=section.text, metadata=metadata))

    chunks = SPLITTER.split_documents(pages)
    if not chunks:
        raise ExtractionError("No readable text was found in this file.")
    if len(chunks) > MAX_FILE_CHUNKS:
        raise LimitExceeded(
            f"This file is too long: it makes {len(chunks)} passages and the limit is "
            f"{MAX_FILE_CHUNKS}. Split it into smaller files and upload them separately."
        )
    summary = {
        "name": name,
        "chunks": len(chunks),
        "extraction": extracted.method,
        "warnings": extracted.warnings,
    }
    return chunks, summary


def _kept(current: Optional[SearchIndex], new_chunks: list[Document]) -> list[int]:
    """Positions of the current chunks that survive, i.e. aren't from files being replaced."""
    if current is None:
        return []
    replaced = {c.metadata["source"] for c in new_chunks}
    return [i for i, c in enumerate(current.chunks) if c.metadata["source"] not in replaced]


def _check_limits(current: Optional[SearchIndex], new_chunks: list[Document]) -> None:
    kept = [current.chunks[i] for i in _kept(current, new_chunks)]
    documents = {c.metadata["source"] for c in kept + new_chunks}
    if len(documents) > MAX_DOCUMENTS:
        raise LimitExceeded(f"A workspace can hold up to {MAX_DOCUMENTS} documents. Remove one to add another.")
    total = len(kept) + len(new_chunks)
    if total > MAX_WORKSPACE_CHUNKS:
        raise LimitExceeded(
            f"That would bring the workspace to {total} passages, over the limit of "
            f"{MAX_WORKSPACE_CHUNKS}. Remove a document to make room."
        )


def _merge(
    current: Optional[SearchIndex],
    new_chunks: list[Document],
    new_embeddings: np.ndarray,
) -> tuple[list[Document], np.ndarray]:
    """Adds chunks to an index, replacing older chunks from the same files."""
    _check_limits(current, new_chunks)
    keep = _kept(current, new_chunks)
    kept_chunks = [current.chunks[i] for i in keep] if current else []
    kept_embeddings = current.embeddings[keep] if current else new_embeddings[:0]

    next_id = 1 + max(
        (int(c.metadata["chunk_id"].removeprefix("chunk_")) for c in kept_chunks),
        default=-1,
    )
    renumbered = [
        Document(page_content=c.page_content, metadata={**c.metadata, "chunk_id": f"chunk_{i}"})
        for i, c in enumerate(new_chunks, start=next_id)
    ]
    return kept_chunks + renumbered, np.vstack([kept_embeddings, new_embeddings])


def add_files(workspace: str, files: list[tuple[str, bytes]]) -> list[dict]:
    """Indexes (filename, contents) pairs into a workspace. Returns a summary per file."""
    prune_expired_workspaces()

    chunks, summaries = [], []
    for name, data in files:
        file_chunks, summary = _chunk_file(name, data)
        chunks.extend(file_chunks)
        summaries.append(summary)

    # Check limits before spending time on embeddings.
    _check_limits(load_workspace(workspace, fresh=True), chunks)

    embeddings = embed([c.page_content for c in chunks])
    # Embedding can take a while, so re-read in case the workspace changed meanwhile.
    merged_chunks, merged_embeddings = _merge(load_workspace(workspace, fresh=True), chunks, embeddings)
    save_workspace(workspace, merged_chunks, merged_embeddings)
    return summaries


def add_sample(workspace: str) -> dict:
    """Copies the bundled sample document into a workspace, reusing its embeddings."""
    sample = sample_index()
    merged_chunks, merged_embeddings = _merge(
        load_workspace(workspace, fresh=True), sample.chunks, sample.embeddings
    )
    save_workspace(workspace, merged_chunks, merged_embeddings)
    name = sample.chunks[0].metadata["source"]
    return {"name": name, "chunks": len(sample.chunks), "extraction": None, "warnings": []}


def remove_document(workspace: str, name: str) -> bool:
    current = load_workspace(workspace, fresh=True)
    if current is None:
        return False
    keep = [i for i, c in enumerate(current.chunks) if c.metadata["source"] != name]
    if len(keep) == len(current.chunks):
        return False
    save_workspace(workspace, [current.chunks[i] for i in keep], current.embeddings[keep])
    return True


def describe(index: Optional[SearchIndex]) -> list[dict]:
    if index is None:
        return []
    counts = Counter(c.metadata["source"] for c in index.chunks)
    methods = {c.metadata["source"]: c.metadata.get("extraction") for c in index.chunks}
    return [{"name": name, "chunks": count, "extraction": methods[name]} for name, count in counts.items()]


def build_sample(paths: list[Path]) -> int:
    chunks = []
    for path in paths:
        chunks.extend(_chunk_file(path.name, path.read_bytes())[0])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
    SAMPLE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_INDEX_PATH.write_bytes(_serialize(chunks, embed([c.page_content for c in chunks])))
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the bundled sample index.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(f"Indexed {build_sample(args.paths)} chunks.")
