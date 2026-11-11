"""
ONNX Runtime versions of the embedding and reranking models. Sessions load
lazily, so a cold start only pays for them on the first request that needs
them.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import EMBEDDING_MODEL_DIR, RERANKER_MODEL_DIR

MAX_TOKENS = 512
EMBED_BATCH_SIZE = 16


@lru_cache(maxsize=None)
def _load(model_dir: Path):
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding()
    session = ort.InferenceSession(
        str(model_dir / "model_quantized.onnx"), providers=["CPUExecutionProvider"]
    )
    return tokenizer, session


def _run(model_dir: Path, inputs: list) -> np.ndarray:
    tokenizer, session = _load(model_dir)
    encodings = tokenizer.encode_batch(inputs)
    return session.run(
        None,
        {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        },
    )[0]


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length BGE embeddings (CLS pooling), one row per text."""
    batches = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = _run(EMBEDDING_MODEL_DIR, texts[start : start + EMBED_BATCH_SIZE])
        batches.append(batch[:, 0])
    vectors = np.concatenate(batches)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def rerank(query: str, passages: list[str]) -> np.ndarray:
    """Cross-encoder relevance logits, one per passage. Higher is more relevant."""
    return _run(RERANKER_MODEL_DIR, [(query, passage) for passage in passages])[:, 0]
