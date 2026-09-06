import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_PERSIST_DIR = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# FIX: Use a currently active and fully supported Groq model
LLM_MODEL = "openai/gpt-oss-20b" 
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"