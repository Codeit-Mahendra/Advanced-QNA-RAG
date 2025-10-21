"""
Advanced QnA RAG System - FastAPI Backend
This application implements a Retrieval-Augmented Generation (RAG) system
that combines document retrieval with LLM generation for intelligent Q&A.
"""

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

# =============================================================================
# CONFIGURATION & INITIALIZATION
# =============================================================================

# Load environment variables from .env file for secure configuration
load_dotenv()

# Get API keys from environment variables (secure way to handle credentials)
# In production, these would be set in the deployment environment (like Render)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # For vector database
GROQ_API_KEY = os.getenv("GROQ_API_KEY")          # For LLM API access

# Validate that required environment variables are present
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing required environment variables: PINECONE_API_KEY and GROQ_API_KEY")

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Advanced QnA RAG System", 
    version="1.0.0"
)

# =============================================================================
# MIDDLEWARE SETUP
# =============================================================================

# Add CORS middleware to allow cross-origin requests
# Essential for frontend-backend communication when they're on different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

# Serve static files (HTML, CSS, JS) from the 'static' directory
# This makes the frontend accessible at the root URL
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# RAG SYSTEM INITIALIZATION
# =============================================================================

# Global variable to hold our RAG chain
rag_chain = None

try:
    # 1. LOAD EMBEDDINGS MODEL
    # This converts text into numerical vectors for similarity search
    embeddings = download_embeddings()
    
    # 2. CONNECT TO VECTOR DATABASE (Pinecone)
    # Pinecone stores and retrieves document embeddings efficiently
    index_name = "advanced-qna-rag"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # 3. CREATE RETRIEVER
    # Configures how to search for relevant documents
    retriever = docsearch.as_retriever(
        search_type="similarity",  # Use cosine similarity
        search_kwargs={"k": 5}     # Return top 5 most relevant documents
    )

    # 4. INITIALIZE LLM (Groq for fast inference)
    chatModel = ChatGroq(
        model="qwen/qwen3-32b",    # Specific model choice for quality/performance
        temperature=0.3,           # Controls randomness (0=deterministic, 1=creative)
        groq_api_key=GROQ_API_KEY
    )
    print(f"Using model: {chatModel.model_name}")

    # 5. SETUP PROMPT TEMPLATE
    # system_prompt (imported from src.prompt) contains instructions for the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # System role: sets behavior/context
        ("human", "{input}"),       # Human role: user's question
    ])

    # 6. CREATE PROCESSING CHAINS
    # Document chain: handles how to process retrieved documents
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    
    # RAG chain: combines retrieval + generation in one pipeline
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("✅ RAG system initialized successfully!")
    
except Exception as e:
    # Graceful error handling - system can still run but without RAG functionality
    print(f"❌ Error initializing RAG system: {e}")
    rag_chain = None

# =============================================================================
# RESPONSE PROCESSING UTILITIES
# =============================================================================

def clean_response(text: str) -> str:
    """
    Post-process LLM responses to remove unwanted content and formatting.
    This is crucial because LLMs sometimes include internal thinking or formatting.
    """
    # Remove <think> tags and all content between them (internal reasoning)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any other XML-like tags that might be present
    text = re.sub(r'<.*?>', '', text)
    
    # Remove common internal reasoning phrases that might leak through
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
    
    # Clean up extra whitespace and newlines for better readability
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    # Remove any remaining contextual phrases at the start of response
    text = re.sub(r'^(Based on|According to|In the context).*?,\s*', '', text, flags=re.IGNORECASE)
    
    # Character replacements for consistency
    text = text.replace('@', '-')
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    
    return text

# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve the main chat interface.
    Returns the HTML file that contains the frontend chat application.
    """
    return FileResponse("static/chat.html")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for deployment platforms (like Render).
    Used by load balancers and monitoring systems to verify service availability.
    """
    return {"status": "healthy", "service": "Advanced QnA RAG"}

@app.post("/get")
async def chat(msg: str = Form(...)):
    """
    Main chat endpoint that processes user questions through the RAG pipeline.
    
    Flow:
    1. User question → 2. Document retrieval → 3. LLM generation → 4. Response cleaning
    
    Args:
        msg: The user's question from the form data
    
    Returns:
        JSON response with the cleaned answer or error message
    """
    # Check if RAG system is properly initialized
    if rag_chain is None:
        return JSONResponse(
            content={"answer": "Service is currently initializing. Please try again in a moment."}
        )
    
    try:
        # Log the input for debugging and monitoring
        print(f"Input: {msg}")
        
        # Execute the full RAG pipeline
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        
        # Clean the response to remove unwanted formatting/thinking
        cleaned_answer = clean_response(answer)
        
        # Log the response for debugging
        print(f"Response: {cleaned_answer}")
        
        # Return the cleaned answer to the user
        return JSONResponse(content={"answer": cleaned_answer})
        
    except Exception as e:
        # Handle any errors during processing
        print(f"Error processing request: {e}")
        return JSONResponse(
            content={"answer": "Sorry, I encountered an error processing your request. Please try again."},
            status_code=500
        )

# =============================================================================
# DEVELOPMENT SERVER CONFIGURATION
# =============================================================================

if __name__ == '__main__':
    # Get port from environment variable (for deployment platforms) or default to 8080
    port = int(os.environ.get("PORT", 8080))
    
    # Import uvicorn here to avoid dependency issues
    import uvicorn
    
    # Start the FastAPI server
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all network interfaces
        port=port, 
        debug=False      # Debug mode off for production
    )