"""
DATA INGESTION PIPELINE FOR RAG SYSTEM
This script handles the complete process of loading PDF documents, 
processing them, and storing them in Pinecone vector database.
"""

from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Load environment variables from .env file for secure credential management
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # For vector database operations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")          # For LLM API (future use)

# Set them in environment variables (optional but ensures availability for other libraries)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# =============================================================================
# DOCUMENT PROCESSING PIPELINE
# =============================================================================

# 1. LOAD PDF FILES
# Loads and extracts text content from PDF files in the specified directory
extracted_data = load_pdf_files("data")
# Output: Raw text extracted from all PDF documents

# 2. FILTER DOCUMENTS
# Processes raw extracted data to create minimal, clean document objects
minimal_docs = filter_to_minimal_docs(extracted_data)
# Purpose: Removes noise, empty documents, and standardizes document structure

# 3. TEXT CHUNKING
# Splits documents into smaller chunks for efficient retrieval
texts_chunk = text_split(minimal_docs)
# Why chunking is important:
# - Prevents context window overflow in LLMs
# - Enables more precise retrieval of relevant information
# - Improves answer quality by focusing on specific content

# 4. INITIALIZE EMBEDDINGS
# Loads the embedding model that converts text to numerical vectors
embedding = download_embeddings()
# Embeddings transform semantic meaning into vector space for similarity search

# =============================================================================
# VECTOR DATABASE SETUP
# =============================================================================

# Initialize Pinecone client with API key
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# Define index name (consistent with the main application)
index_name = "advanced-qna-rag"

# Check if index exists by listing all available indexes
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}")
    
    # Create new vector index with specific configuration
    pc.create_index(
        name=index_name,
        dimension=768,      # Dimension of the embedding vectors
        # dimension=384,    # Alternative dimension for smaller models
        metric="cosine",    # Similarity metric (cosine = angle between vectors)
        spec=ServerlessSpec(
            cloud="aws",    # Cloud provider
            region="us-east-1"  # Geographic region
        )
    )
    print(f"Index '{index_name}' created successfully!")
    
    # Why these parameters matter:
    # - dimension: Must match embedding model output (768 for all-mpnet-base-v2)
    # - metric: Cosine similarity works well for semantic search
    # - serverless: No infrastructure management required
    
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index (whether newly created or existing)
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# =============================================================================
# VECTOR STORE POPULATION
# =============================================================================

# Create vector store by uploading document chunks with their embeddings
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,      # Processed document chunks
    embedding=embedding,        # Embedding model instance
    index_name=index_name       # Target Pinecone index
)
print("Vector store created successfully!")

# What happens here:
# 1. Each document chunk is converted to embedding vector
# 2. Vectors are uploaded to Pinecone index
# 3. Metadata (source, position) is stored with each vector
# 4. Index becomes searchable for semantic similarity queries