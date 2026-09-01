# 📄 Document Intelligence Chatbot (RAG System)

An AI-powered Document Intelligence Chatbot that answers questions using information retrieved directly from a text document.

## 🎯 Project Goal

The goal of this project is to build a Retrieval-Augmented Generation (RAG) system that can search relevant information from a document and provide context-based answers using an LLM.

## 🛠️ Tech Stack

- Python
- Google Gemini API
- FAISS
- Sentence Transformers
- Python-dotenv

## ⚙️ How It Works

1. Loads text from `document.txt`
2. Splits the document into smaller chunks
3. Converts chunks into embeddings
4. Stores embeddings in a FAISS vector index
5. Converts the user's question into an embedding
6. Finds the most relevant document chunks
7. Sends the retrieved context to Gemini
8. Generates an answer based on the document

## 📁 Project Structure

```text
document-intelligence-rag-chatbot/
│
├── rag_chatbot.py
├── document.txt
├── .gitignore
└── README.md
