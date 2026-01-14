---
title: Intelligent Research Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Intelligent Research Assistant

An AI-powered research assistant that allows you to upload documents (PDFs, Web pages, YouTube transcripts) and have natural conversations about them using RAG (Retrieval-Augmented Generation).

## Features

- **Multi-source Ingestion**: 
  - **PDFs**: Research papers, books, and reports.
  - **Web pages**: Articles and documentation.
  - **YouTube videos**: Automatic transcript extraction.
- **Natural Conversations**: Ask questions and get detailed answers based on your uploaded content.
- **Citations & Sources**: The AI shows exactly which documents were used to generate the answer.
- **Conversation Memory**: Remembers chat history for context-aware follow-up questions.

## How It Works (RAG Architecture)

1. **Document Processing**: Documents are broken into chunks and converted into mathematical representations (embeddings) using Azure OpenAI's `text-embedding-3-large` model.
2. **Storage**: Embeddings are stored in **ChromaDB**, enabling fast semantic search.
3. **Question Answering**: 
    - Questions are converted to embeddings.
    - Relevant document chunks are retrieved.
    - **gpt-oss-120b** (OpenAI's open-weight model) generates an answer based on retrieved chunks.
    - Citations are added to show sources.
4. **Conversation Memory**: Context is maintained throughout the session.

## Tech Stack

- **LLM**: gpt-oss-120b (117 billion parameters)
- **Embeddings**: text-embedding-3-large (3072 dimensions)
- **UI Framework**: Gradio
- **Vector Database**: ChromaDB
- **Orchestration**: LangChain
- **Deployment**: Azure OpenAI

## Setup

1. **Clone the repository**
2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment**:
   Create a `.env` file with your Azure OpenAI/OpenAI credentials:
   ```env
   OPENAI_API_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   ```
5. **Run the application**:
   ```bash
   python app.py
   ```

## Why This Project?

This project demonstrates production-ready RAG implementation, semantic search optimization, and conversational AI engineering.
