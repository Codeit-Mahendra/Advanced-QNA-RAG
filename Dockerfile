FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++

COPY requirements.txt .

RUN pip install --upgrade pip

# Install packages one by one to avoid dependency conflicts
RUN pip install fastapi uvicorn python-multipart
RUN pip install langchain langchain-community langchain-core
RUN pip install langchain-groq langchain-pinecone
RUN pip install pinecone-client sentence-transformers
RUN pip install groq pypdf python-dotenv pydantic requests numpy

COPY . .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]