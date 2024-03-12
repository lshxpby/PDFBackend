# PDFBackend

PDFBackend is a FastAPI-based backend service for extracting text from PDF files and providing conversational AI capabilities through language models and retrieval-based chat systems. It utilizes FastAPI for creating APIs, PyMuPDF (fitz) for reading PDF files, and integrates various components from the langchain library for natural language processing tasks.

## Getting Started

To get started with PDFBackend, follow these steps:

1. Create a `.env` file in the root directory of the project.
2. Paste the following line into the `.env` file to set up your Hugging Face API token:
HUGGINGFACEHUB_API_TOKEN=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

3. Ensure you have Python 3.9.13 installed.

## Installation

To install PDFBackend and its dependencies, use the following command:

```bash
pip install fastapi pymupdf python-dotenv transformers faiss-ct langchain==2.2.2 uvicorn
```
You can run the Backend Service using the following command

```uvicorn main:app --reload```
