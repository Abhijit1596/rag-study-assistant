# 📚 RAG Study Assistant

A Retrieval-Augmented Generation (RAG) based study assistant built using Python, Streamlit, FAISS, Sentence Transformers, and Groq LLM.

## Features

- Upload one or multiple PDF files
- Generate detailed PDF summaries
- Generate important exam-oriented questions
- Ask questions directly from uploaded PDFs
- Semantic search using vector embeddings
- Fast responses using Groq LLM

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Groq API
- PyPDF2

## Project Structure

```
RAG_Study_Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
│
└── src/
    ├── chatbot.py
    ├── embeddings.py
    └── pdf_processor.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Abhijit1596/rag-study-assistant.git
cd rag-study-assistant
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Groq API key:

```env
GROQ_API_KEY=YOUR_API_KEY
```

Run the application:

```bash
streamlit run app.py
```

## How It Works

1. User uploads PDF files.
2. Text is extracted from PDFs.
3. Text is converted into vector embeddings.
4. FAISS creates a vector database.
5. Relevant chunks are retrieved for queries.
6. Groq LLM generates answers, summaries, and important questions.

## Author

Abhijit Borah

IIT Guwahati – BSc Data Science and AI
