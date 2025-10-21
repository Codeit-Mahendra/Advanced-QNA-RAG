"""
DATA INGESTION PIPELINE FOR RAG SYSTEM
Loads PDF documents, processes them, and stores in Pinecone.
"""

from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "advanced-qna-rag"

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
documents = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(documents)
chunks = text_split(minimal_docs)
embedding = download_embeddings()
PineconeVectorStore.from_documents(chunks, embedding=embedding, index_name=index_name)

print("Vector store created.")
