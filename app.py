

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings and vector store
embeddings = download_embeddings()

index_name = "advanced-qna-rag"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 7})

chatModel = ChatGroq(
    model="groq/compound",  # Using a commonly available Groq model
    temperature=0.2,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Initialize FastAPI app
app = FastAPI(title="advanced-qna-rag", debug=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the static HTML file directly
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/c.html")

@app.post("/get")
async def chat(msg: str = Form(...)):
    print(f"Input: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response['answer']}")
    return JSONResponse(content={"answer": response["answer"]})

# Alternative JSON endpoint
@app.post("/api/chat")
async def chat_api(request: dict):
    msg = request.get("msg", "")
    if not msg:
        return JSONResponse(content={"error": "No message provided"}, status_code=400)
    
    print(f"Input: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response['answer']}")
    return {"answer": response["answer"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)