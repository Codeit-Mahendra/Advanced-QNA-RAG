from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # ✅ This line was missing

# Set them in environment (optional, if needed by other libraries)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_files("D:\Advanced-QNA-RAG\data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

####--------------------------------------------------------------

index_name = "advanced-qna-rag"

# Check if index exists by listing all indexes
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

####----------------------------------------------------------------

# Assuming texts_chunk and embedding are defined from earlier cells
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,  # Note: Your notebook uses 'embeddings' in cell 31, which should be 'embedding'
    index_name=index_name
)
print("Vector store created successfully!")