import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Generator drafts answers; a separate, stronger judge model audits them, so
# the model grading an answer is never the one that wrote it.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

# Quantized ONNX exports of BAAI/bge-small-en-v1.5 and
# cross-encoder/ms-marco-MiniLM-L-6-v2. Bundled with the deployment so that
# no torch install or model download is needed at runtime.
EMBEDDING_MODEL_DIR = BACKEND_DIR / "models" / "bge-small-en-v1.5"
RERANKER_MODEL_DIR = BACKEND_DIR / "models" / "ms-marco-MiniLM-L-6-v2"

# Models that turn non-text uploads into text: images (and scanned PDF pages)
# are read by a vision model, and audio or video is transcribed.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")

# The sample document ships prebuilt with the deployment and is read-only.
SAMPLE_INDEX_PATH = BACKEND_DIR / "index" / "index.npz"
SAMPLE_NAME = "attentionisalluneed.pdf"

# Each visitor gets a private workspace with its own index. On Vercel the
# filesystem is read-only, so workspaces are saved to Vercel Blob.
LOCAL_WORKSPACE_DIR = BACKEND_DIR / "index" / "workspaces"
BLOB_WORKSPACE_PREFIX = "hyrag/workspaces/"
BLOB_ENABLED = bool(os.getenv("BLOB_READ_WRITE_TOKEN") or os.getenv("BLOB_STORE_ID"))
INDEX_REFRESH_SECONDS = 30
# Workspaces are deleted this many days after their last change.
WORKSPACE_TTL_DAYS = 7

ON_VERCEL = bool(os.getenv("VERCEL"))
UPLOADS_ENABLED = not ON_VERCEL or BLOB_ENABLED

# Vercel rejects request bodies above 4.5 MB.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
# Embedding runs on the function's CPU, so these bound how long an upload takes.
MAX_FILE_CHUNKS = 300
MAX_WORKSPACE_CHUNKS = 1500
MAX_DOCUMENTS = 25
# Each scanned PDF page costs one vision model call, and Groq's free tier
# allows about five a minute.
MAX_OCR_PAGES = 5
