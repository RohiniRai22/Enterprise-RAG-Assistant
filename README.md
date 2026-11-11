# Enterprise RAG Assistant 🤖

An AI-powered Enterprise Retrieval-Augmented Generation (RAG) application that answers questions from company documents using semantic search and Google Gemini.

## 🚀 Features

- Supports PDF, DOCX, and TXT documents.
- Recursive text chunking.
- Semantic search using Sentence Transformers.
- FAISS vector database for similarity search.
- Google Gemini for AI-powered answer generation.
- Department-based filtering (HR, Finance, IT).
- Source document references with similarity scores.
- RAG evaluation using RAGAS.
- Interactive Streamlit interface.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM:** Google Gemini
- **Embeddings:** Sentence Transformers
- **Vector Database:** FAISS
- **Evaluation:** RAGAS
- **Document Processing:** PyMuPDF, python-docx

## 🏗️ Architecture

```text
Company Documents
       ↓
Document Loader
       ↓
Text Chunking
       ↓
Embedding Generation
       ↓
FAISS Vector Store
       ↓
User Query
       ↓
Semantic Retrieval
       ↓
Relevant Context
       ↓
Google Gemini
       ↓
AI Generated Answer
       ↓
Streamlit Interface
```

## 📂 Project Structure

```text
Enterprise-RAG-Assistant/
│
├── app.py
├── app/
├── src/
├── evaluation/
├── build_vector_store.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/RohiniRai22/Enterprise-RAG-Assistant.git
