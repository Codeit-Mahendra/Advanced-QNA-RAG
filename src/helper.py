
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
warnings.filterwarnings("ignore")

def load_pdf_files(data):

    loader = DirectoryLoader(  #Created to load all PDF files from a directory
        data,
        glob = "*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()
    return documents
print("PDF files loaded successfully!")


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:  # Create a function to filter documents
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs
print("Minimal document loaded successfully!")

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(  # defined to split text into smaller chunks
        chunk_size=300,
        chunk_overlap=40,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        length_function=len
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk
print("Text splitting successful!")

def download_embeddings():
    embeddings = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2"
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return embeddings

# Test it
embeddings = download_embeddings()
print("Embeddings loaded successfully!")

