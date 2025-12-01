# Note-Reviewer
A RAG project purposely designed to read documents and answer user questions pertaining to the document.

## Project Architecture

```text
[Your Papers] -> [Chunker] -> [Chunks] -> [Embeddings (MiniLM)] -> [FAISS Vector Store]
                                         ↑
                             [Your Question] -> [Retriever] -> [Top Chunks]
                                                             |
                                                             v
                                                    [Prompt Constructor]
                                                             |
                                                             v
                                                          [Local LLM]
                                                             |
                                                             v
                                                         [Answer/UI]
```

## Tools/Packages used

- PDF Loader - PyPDF2
- Chunking - Custom Python Splitter
- Embeddings - MiniLM from Sentence-Transformers (all-MiniLM-L6-v2)
- Vector Database Store - FAISS (Good for local, free, simple vector search)
- LLM Inference - Hugging Face Local Models (Ollama)
- UI - Gradio (Testing RAG system)

## Installation Guide

Python 3.11.3 is recommended to run this project.
It is recommended to create a virtual environment before working/using this project. 
After you create your virtual environment, run 'python3 -m pip install -r requirements.txt' to install the required python modules.

# Usage

To run the app, use 'python -m app'. A URL should appear in the terminal which you can access locally.