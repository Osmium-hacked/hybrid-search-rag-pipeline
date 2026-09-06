import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import nltk
from .config import CHROMA_PERSIST_DIR, BM25_INDEX_PATH, EMBEDDING_MODEL

# Ensure nltk tokenizer is available for BM25
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def ingest_documents(file_paths: list[str]):
    documents = []
    
    # 1. Multi-format Loader
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt") or path.endswith(".md"):
            loader = TextLoader(path)
        else:
            continue
        documents.extend(loader.load())

    # 2. Configurable Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    # 3. Dense Indexing (Local HuggingFace Embeddings)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectorstore.persist()

    # 4. Sparse Indexing (BM25)
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return len(chunks)