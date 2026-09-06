from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .config import GROQ_API_KEY, LLM_MODEL

llm = ChatGroq(model_name=LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.1)

def generate_answer(query: str, retrieved_chunks: list):
    context_blocks = []
    for chunk in retrieved_chunks:
        cid = chunk.metadata.get('chunk_id', 'unknown')
        context_blocks.append(f"[Source: {cid}]\n{chunk.page_content}")
    
    context_str = "\n\n---\n\n".join(context_blocks)
    
    sys_prompt = (
        "You are an expert technical assistant. Answer the user's question based ONLY on the provided context.\n"
        "You MUST cite your claims using the exact source brackets provided, e.g., [Source: chunk_1].\n"
        "If the context does not contain the answer, explicitly state 'I do not have enough information to answer this.' Do not hallucinate."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    
    # FIX: Added .invoke() here
    response = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return response.content, context_str

def verify_citations(query: str, answer: str, context: str):
    eval_prompt = (
        f"Verify if the following answer is fully supported by the provided context.\n"
        f"Question: {query}\nContext: {context}\nAnswer: {answer}\n\n"
        f"Respond with 'VERIFIED' if all claims have valid citations, or 'HALLUCINATED' if claims lack context support."
    )
    
    # FIX: Added .invoke() here
    eval_response = llm.invoke([HumanMessage(content=eval_prompt)])
    return eval_response.content