# Personal Transaction AI Assistant 💳🤖

An AI-powered Personal Transaction Assistant that uses a **RAG (Retrieval-Augmented Generation)** pipeline to let users query and analyze their financial data using natural language.

## 🚀 Features

- Ask questions about your transactions in plain English
- RAG-based context retrieval for accurate answers
- Real-time financial insights
- Spending analysis by category
- Duplicate transaction detection
- Clean and responsive UI

## 🧠 How it Works

1. Transactions are stored and embedded
2. FAISS retrieves relevant transaction context
3. LLM generates intelligent responses using retrieved data
4. FastAPI backend serves results to React frontend

## 🛠️ Tech Stack

- Frontend: React
- Backend: FastAPI
- AI: LLM (Groq / OpenAI compatible)
- RAG: FAISS + embeddings (sentence-transformers)
- Vector DB: FAISS
- LangChain

## 📌 Use Cases

- Personal finance assistant
- Banking analytics tool
- Smart financial dashboards
- AI-powered expense tracking

## ⚡ Getting Started

```bash
# Backend
python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Frontend
npm install
npm run dev
```

## Scope of Improvements - 

- SelfQueryRetreiver to be configured and used in place of the MMR retriever which works for small number of cases, but as the number of transactions increase, the Self Query Retriever will work more efficiently handling the retrieval process more smoothly through referring the metadata of the transactions data instead of giving the top 'k' similarity searches.

- Rate limiting on server-side

- Input sanitization to the chatbot