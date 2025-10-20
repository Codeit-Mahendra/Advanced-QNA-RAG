# Advanced QnA RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built for question-answering tasks, specifically implemented to analyze and answer questions about literary content.

![RAG System Interface](image.png)

## 📖 Project Overview

This project implements an advanced RAG pipeline that can answer detailed questions about textual content. The current implementation is configured to analyze "The Blue Umbrella" story, providing accurate, context-aware responses to user queries.

## 🚀 Features

- **Advanced Retrieval**: Uses Pinecone vector database for efficient semantic search
- **Groq Integration**: Leverages high-performance LLMs via Groq API
- **FastAPI Backend**: Robust and scalable web API
- **Clean UI**: Simple and intuitive chat interface
- **Context-Aware Responses**: Provides answers based strictly on provided context
- **Response Cleaning**: Advanced post-processing to remove internal model reasoning

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **Vector Database**: Pinecone
- **LLM**: Groq API (Qwen3-32B model)
- **Embeddings**: Sentence Transformers
- **Frontend**: HTML, JavaScript
- **Environment**: dotenv for configuration

## 📋 Setup Instructions

### Prerequisites

- Python 3.8+
- Pinecone API key
- Groq API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Codeit-Mahendra/Advanced-QNA-RAG.git
   cd Advanced-QNA-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Configure Pinecone Index**
   - Ensure you have a Pinecone index named `advanced-qna-rag`
   - The system will automatically load and use this index

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:8080`

## 🎯 Sample Questions to Test

### Character-Based Questions
- **What did Binya exchange to get the blue umbrella?**
- **Who is Binya?**
- **Who is Bijju?**
- **Who gives Binya that pendant?**

### Plot and Setting Questions
- **Tell me about Ram's shop?**
- **How did the wind cause Binya's umbrella to end up in danger?**
- **Where did this Silver pendant come from?**

### Analytical Questions
- **Why did Ram Bharosa make an offer to buy the umbrella, and what does his final reaction suggest about his feelings towards it?**
- **According to Ram Bharosa, what is the umbrella's primary purpose, and what does Binya actually value it for?**

### General Questions
- **Who is the author of this book?**
- **What is the overall summary of the book?**

## 🔧 Technical Architecture

### Components

1. **Vector Store**: Pinecone for efficient similarity search
2. **Embeddings**: Sentence transformers for text representation
3. **LLM**: Groq with Qwen3-32B model for generation
4. **Retrieval Chain**: LangChain for orchestration
5. **API Layer**: FastAPI for web interface
6. **Response Cleaning**: Custom post-processing pipeline

### Key Features

- **Semantic Search**: k=5 similarity search with cosine similarity
- **Temperature Control**: 0.3 for consistent responses
- **Response Cleaning**: Removes internal model reasoning tags
- **Error Handling**: Robust error management throughout the pipeline

## 📁 Project Structure

```
Advanced-QNA-RAG/
├── app.py                 # Main FastAPI application
├── src/
│   ├── helper.py         # Embedding and utility functions
│   └── prompt.py         # System prompts and templates
├── static/
│   └── chat.html         # Web interface
├── requirements.txt      # Python dependencies
└── .env                 # Environment variables
```

## 🎮 Usage

1. Start the application using `python app.py`
2. Open the web interface in your browser
3. Type your question in the input field
4. Receive context-aware answers based on the stored knowledge

## 🔍 How It Works

1. **Question Input**: User submits a question through the web interface
2. **Semantic Search**: System finds the most relevant context chunks from Pinecone
3. **Context Augmentation**: Relevant context is combined with the question
4. **LLM Generation**: Groq model generates an answer based on the augmented prompt
5. **Response Cleaning**: Internal reasoning tags are removed from the response
6. **Output**: Clean, direct answer is displayed to the user

## ⚙️ Configuration

### Model Settings
- **Model**: `qwen/qwen3-32b`
- **Temperature**: 0.3
- **Search Type**: Similarity
- **Top K**: 5 documents

### Vector Store
- **Index**: `advanced-qna-rag`
- **Embeddings**: Sentence transformers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Groq for high-performance LLM inference
- Pinecone for vector database services
- LangChain for the RAG framework
- FastAPI for the web framework

---

**Note**: Make sure to set up your API keys in the `.env` file before running the application. The system is currently configured for "The Blue Umbrella" story but can be adapted for other texts by updating the Pinecone index.





