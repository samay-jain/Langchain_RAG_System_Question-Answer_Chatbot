# LangChain Q&A Chatbot with OpenAI & Ollama

This project demonstrates how to build a question-answering chatbot that extracts real content from websites, stores it using vector embeddings (ChromaDB), and answers natural language queries. Users can choose between using OpenAI's GPT models or running local inference with Ollama (e.g., LLaMA 3.2). Ideal for learning retrieval-augmented generation (RAG), vector databases, and multi-LLM integration using LangChain.

---

## 🔧 Features

- ✅ Web scraping using Selenium
- ✅ Text chunking for better embedding
- ✅ Vector store with ChromaDB
- ✅ Embeddings via OpenAI or Ollama (`nomic-embed-text`)
- ✅ Natural language question answering via `gpt-4o-mini` or `llama3.2`
- ✅ Source citation for retrieved answers
- ✅ Command-line interface for interaction

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Example Use

Enter your question: What is an NFT?

Response:
An NFT (Non-Fungible Token) is a unique digital asset representing ownership of a specific item, such as art, music, or collectibles, secured via blockchain...

Sources:
- https://beebom.com/what-is-nft-explained/

---
