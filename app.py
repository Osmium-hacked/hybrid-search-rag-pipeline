import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Enterprise RAG System")
st.sidebar.header("Document Ingestion")

# File path input (For local testing, pointing to a local directory path)
doc_path = st.sidebar.text_input("Enter local file path to ingest (e.g., ./docs/sample.txt)")
if st.sidebar.button("Ingest Document"):
    with st.spinner("Chunking, Embedding, and Indexing..."):
        res = requests.post(f"{API_URL}/v1/ingest", json={"file_paths": [doc_path]})
        if res.status_code == 200:
            st.sidebar.success(f"Indexed {res.json()['chunks_indexed']} chunks.")
        else:
            st.sidebar.error("Ingestion failed.")

# Q&A Interface
query = st.text_input("Ask a question about your documents:")
if st.button("Submit"):
    with st.spinner("Running Hybrid Search & Generation..."):
        # Capture the raw response first
        response = requests.post(f"{API_URL}/v1/ask", json={"query": query})
        
        # Check if the backend succeeded (HTTP 200)
        if response.status_code == 200:
            res = response.json()
            
            st.markdown("### Answer")
            st.write(res["answer"])
            
            st.markdown("### Confidence Metrics")
            if res["citations_verified"]:
                st.success("✅ Citations Verified by LLM-Judge")
            else:
                st.warning("⚠️ Warning: Hallucination detected in citations.")
                
            st.markdown("### Retrieved Chunks (RRF + Reranked)")
            st.json(res["retrieved_sources"])
        else:
            # Display the actual error on the UI instead of crashing
            st.error(f"Backend API Error: {response.text}")