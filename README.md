# LangChain Q&A Chatbot with OpenAI & Ollama

This project demonstrates how to build a document-based Question Answering (QA) chatbot using [LangChain](https://www.langchain.com/), integrated with either **OpenAI** or **Ollama** for the LLM backend.

It scrapes and indexes content from URLs and enables conversational Q&A using vector search + LLM.

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

