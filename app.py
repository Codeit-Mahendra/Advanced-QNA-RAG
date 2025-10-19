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

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set them in environment (optional, if needed by other libraries)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "advanced-qna-rag"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Update your model to a working one
chatModel = ChatGroq(
    model="groq/compound",  # Working model
    temperature=0.1,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)
print(f"Using model: {chatModel.model_name}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Initialize FastAPI app
app = FastAPI()

# Mount static files only (since chat.html is in static folder)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the chat.html file directly from static folder
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/chat.html")

@app.post("/get")
async def chat(msg: str = Form(...)):
    print(f"Input: {msg}")
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]
    
    # Clean the response - fix common formatting issues
    cleaned_answer = answer.replace('@', '-')  # Replace @ with -
    cleaned_answer = cleaned_answer.replace('\u2013', '-')  # Replace en-dash with regular dash
    cleaned_answer = cleaned_answer.replace('\u2014', '-')  # Replace em-dash with regular dash
    
    print(f"Response: {cleaned_answer}")
    return JSONResponse(content={"answer": cleaned_answer})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, debug=True)