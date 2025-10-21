"""
HELPER FUNCTIONS FOR RAG SYSTEM
This module contains all the utility functions for document loading, processing, 
and embedding generation used in the RAG pipeline.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List
from langchain.schema import Document

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner execution

# =============================================================================
# DOCUMENT LOADING FUNCTIONS
# =============================================================================

def load_pdf_files(data):
    """
    Load all PDF files from a specified directory using LangChain's DirectoryLoader.
    
    Args:
        data (str): Path to the directory containing PDF files
        
    Returns:
        List[Document]: List of Document objects with page content and metadata
    
    Process:
        1. Scans directory for all PDF files
        2. Uses PyPDFLoader to extract text from each page
        3. Returns structured Document objects with metadata
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",           # Pattern to match PDF files
        loader_cls=PyPDFLoader, # Specific loader for PDF format
    )

    documents = loader.load()
    return documents

print("PDF files loaded successfully!")


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to retain only essential metadata for efficient storage and retrieval.
    
    Why this is important:
    - Reduces storage overhead in vector database
    - Improves retrieval performance
    - Maintains source attribution for answer verification
    - Removes unnecessary metadata that doesn't aid retrieval
    
    Args:
        docs (List[Document]): Original documents with full metadata
        
    Returns:
        List[Document]: Documents with only 'source' metadata preserved
    """
    minimal_docs: List[Document] = []
    
    for doc in docs:
        # Extract source information for provenance tracking
        src = doc.metadata.get("source")
        
        # Create new document with only essential metadata
        minimal_docs.append(
            Document(
                page_content=doc.page_content,  # Preserve original content
                metadata={"source": src}        # Keep only source information
            )
        )
    return minimal_docs


# =============================================================================
# TEXT PROCESSING FUNCTIONS
# =============================================================================

def text_split(minimal_docs):
    """
    Split documents into optimal-sized chunks for efficient retrieval and processing.
    
    Chunking Strategy:
    - Recursive character splitting for natural text boundaries
    - Balanced chunk size for context preservation and precision
    - Overlap to maintain context continuity across chunks
    
    Args:
        minimal_docs (List[Document]): Filtered documents to split
        
    Returns:
        List[Document]: Chunked documents ready for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # Optimal chunk size for balance between context and precision
        chunk_size=300,       # Characters per chunk (~50-100 words)
        chunk_overlap=40,     # Overlap to preserve context across chunks
        
        # Hierarchy of separators for natural text splitting
        separators=[
            "\n\n",  # Paragraph breaks (highest priority)
            "\n",    # Line breaks
            ". ",    # Sentences
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semi-colons
            ", ",    # Commas
            " "      # Words (lowest priority)
        ],
        length_function=len   # Use character count for consistency
    )
    
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def download_embeddings():
    """
    Initialize and return a sentence transformer model for text embeddings.
    
    Model Selection Rationale:
    - 'all-mpnet-base-v2': Higher quality, 768 dimensions, better performance
    - 'all-MiniLM-L6-v2': Faster, 384 dimensions, good for production
    
    Embeddings convert semantic meaning into numerical vectors that enable:
    - Semantic similarity search
    - Context understanding
    - Cross-lingual capabilities
    
    Returns:
        HuggingFaceEmbeddings: Initialized embedding model
    """
    embeddings = HuggingFaceEmbeddings(
        # Using higher-quality model for better semantic understanding
        model_name="sentence-transformers/all-mpnet-base-v2"
        
        # Alternative: Faster but less accurate
        # model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

# Test the embedding function
