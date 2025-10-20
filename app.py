from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import re

# Load environment variables
load_dotenv()

# Get API keys from environment (Render will set these)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing required environment variables: PINECONE_API_KEY and GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="Advanced QnA RAG System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components (with error handling)
try:
    embeddings = download_embeddings()
    
    index_name = "advanced-qna-rag"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    chatModel = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )
    print(f"Using model: {chatModel.model_name}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("✅ RAG system initialized successfully!")
    
except Exception as e:
    print(f"❌ Error initializing RAG system: {e}")
    # Set components to None to handle gracefully
    rag_chain = None

def clean_response(text: str) -> str:
    """Remove internal thinking tags and unwanted formatting from the response"""
    # Remove <think> tags and all content between them
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any other XML-like tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove common internal reasoning phrases
    reasoning_phrases = [
        r'Let me think.*?\.',
        r'First,.*?\.', 
        r'Based on.*?context',
        r'According to.*?text',
        r'I should.*?\.',
        r'Looking at.*?\.',
        r'Okay, let me.*?\.',
        r'Let\'s see.*?\.',
        r'The user is asking.*?\.',
        r'I need to look.*?\.'
    ]
    
    for phrase in reasoning_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra whitespace and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    # Remove any remaining "Based on..." or similar phrases at the start
    text = re.sub(r'^(Based on|According to|In the context).*?,\s*', '', text, flags=re.IGNORECASE)
    
    # Character replacements
    text = text.replace('@', '-')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    
    return text

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/chat.html")

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy", "service": "Advanced QnA RAG"}

@app.post("/get")
async def chat(msg: str = Form(...)):
    if rag_chain is None:
        return JSONResponse(content={"answer": "Service is currently initializing. Please try again in a moment."})
    
    try:
        print(f"Input: {msg}")
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        
        cleaned_answer = clean_response(answer)
        
        print(f"Response: {cleaned_answer}")
        return JSONResponse(content={"answer": cleaned_answer})
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(
            content={"answer": "Sorry, I encountered an error processing your request. Please try again."},
            status_code=500
        )

# For local development
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, debug=False)

# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_groq import ChatGroq
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# import re  # ADD THIS IMPORT

# # Load environment variables
# load_dotenv()

# # Get API keys
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Set them in environment (optional, if needed by other libraries)
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# embeddings = download_embeddings()

# index_name = "advanced-qna-rag"
# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# # Update your model to a working one
# chatModel = ChatGroq(
#     # model="llama-3.3-70b-versatile",  # Working model
#     model="qwen/qwen3-32b",  # Working model
#     temperature=0.3,
#     groq_api_key=os.environ.get("GROQ_API_KEY")
# )
# print(f"Using model: {chatModel.model_name}")

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # ADD THE CLEANING FUNCTION HERE
# def clean_response(text: str) -> str:
#     """Remove internal thinking tags and unwanted formatting from the response"""
#     # Remove <think> tags and all content between them
#     text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
#     # Remove any other XML-like tags
#     text = re.sub(r'<.*?>', '', text)
    
#     # Remove common internal reasoning phrases
#     reasoning_phrases = [
#         r'Let me think.*?\.',
#         r'First,.*?\.', 
#         r'Based on.*?context',
#         r'According to.*?text',
#         r'I should.*?\.',
#         r'Looking at.*?\.',
#         r'Okay, let me.*?\.',
#         r'Let\'s see.*?\.',
#         r'The user is asking.*?\.',
#         r'I need to look.*?\.'
#     ]
    
#     for phrase in reasoning_phrases:
#         text = re.sub(phrase, '', text, flags=re.IGNORECASE | re.DOTALL)
    
#     # Clean up extra whitespace and newlines
#     text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newlines
#     text = text.strip()
    
#     # Remove any remaining "Based on..." or similar phrases at the start
#     text = re.sub(r'^(Based on|According to|In the context).*?,\s*', '', text, flags=re.IGNORECASE)
    
#     # Your existing character replacements
#     text = text.replace('@', '-')
#     text = text.replace('\u2013', '-')
#     text = text.replace('\u2014', '-')
    
#     return text

# # Initialize FastAPI app
# app = FastAPI()

# # Mount static files only (since chat.html is in static folder)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Serve the chat.html file directly from static folder
# @app.get("/", response_class=HTMLResponse)
# async def index():
#     return FileResponse("static/chat.html")

# @app.post("/get")
# async def chat(msg: str = Form(...)):
#     print(f"Input: {msg}")
#     response = rag_chain.invoke({"input": msg})
#     answer = response["answer"]
    
#     # USE THE CLEANING FUNCTION HERE - Replace your existing cleaning
#     cleaned_answer = clean_response(answer)
    
#     print(f"Response: {cleaned_answer}")
#     return JSONResponse(content={"answer": cleaned_answer})

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080, debug=True)