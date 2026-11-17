
# 📚 RAG Agent with Citation Grounding

An AI-powered PDF Question Answering Application built using Retrieval-Augmented Generation (RAG).

This application allows users to upload PDF documents, ask questions, and receive AI-generated answers based on the uploaded content, along with source document names and page number citations.

## 🚀 Project Overview

The RAG Agent uses Retrieval-Augmented Generation to find relevant information from PDF documents before generating answers.

Instead of answering from general knowledge, the application retrieves relevant document content and provides it to the AI model as context.

If the answer is not available in the uploaded PDFs, the application responds:

"I couldn't find this information in your PDF."

## ✨ Features

- Upload multiple PDF documents.
- Extract text from PDF files.
- Split documents into smaller text chunks.
- Generate text embeddings.
- Store and search document information using ChromaDB.
- Retrieve relevant information based on user questions.
- Generate answers using a local Ollama language model.
- Display source PDF names and page number citations.
- Basic hallucination control through context-based prompting.
- Chat history.
- Dark-themed Streamlit user interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | User interface |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| Ollama | Local AI language model |
| PyPDF | PDF text extraction |

## 🏗️ Project Architecture

1. User uploads PDF documents.
2. The application extracts text from PDFs.
3. Extracted text is divided into smaller chunks.
4. Sentence Transformers converts chunks into embeddings.
5. ChromaDB stores the embeddings.
6. User enters a question.
7. The question is converted into an embedding.
8. ChromaDB retrieves relevant document chunks.
9. Ollama generates an answer using the retrieved context.
10. The application displays the answer with source citations.

## 📂 Project Structure

```text
Rag-agent-with-citation-grounding/
│
├── documents/
│   └── sample.pdf
│
├── vectorstore/
│
├── main.py
├── app.py
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kavitha123rk/Rag-agent-with-citation-grounding.git
```

### 2. Open the project folder

```bash
cd Rag-agent-with-citation-grounding
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install required libraries

```bash
pip install streamlit pypdf chromadb sentence-transformers ollama
```

### 6. Install Ollama

Download Ollama from:

https://ollama.com/

Pull the required model:

```bash
ollama pull qwen2.5:1.5b
```

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Example Usage

**Example Question:**

What is Elliptic Curve Cryptography?

**Expected Application Behavior:**

- Retrieves relevant information from the uploaded PDF.
- Generates an answer using Ollama.
- Displays the source PDF filename.
- Displays the relevant page number.

**Hallucination Control Test:**

Question: What is Artificial Intelligence?

If this information is not available in the uploaded PDF, the application should respond:

"I couldn't find this information in your PDF."

## 🎯 Project Objective

The main objective is to build a document-based AI question-answering system that improves answer relevance and provides source citations to help users verify the information.

## 🔮 Future Enhancements

- Improve retrieval accuracy.
- Add support for more document formats.
- Improve citation verification.
- Add answer evaluation metrics.
- Improve hallucination detection.
- Deploy the application online.

## 👩‍💻 Developer

**Kavitha K**

Computer Science and Engineering Student

GitHub: https://github.com/kavitha123rk

---

⭐ If you find this project useful, consider giving it a star!
