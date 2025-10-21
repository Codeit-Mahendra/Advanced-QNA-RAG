"""
Advanced QnA RAG System - FastAPI Backend
Implements a Retrieval-Augmented Generation (RAG) system for intelligent Q&A.
"""

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from src.helper import download_embeddings, create_rag_chain
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(title="Advanced QnA RAG System", version="1.0.0")

if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY or PINECONE_API_KEY")

@app.post("/get")
async def chat(query: str = Form(...)):
    try:
        rag_chain = create_rag_chain("advanced-qna-rag")
        response = rag_chain.invoke({"input": query})
        return JSONResponse(content={"answer": response["answer"]})
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            content={"answer": "Error processing request. Try again."},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, debug=False)
