from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

os.environ["HF"] = os.getenv("HF")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = True
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY not found")

# Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Personal Transaction Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# model
LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0
)

# embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# load and format docs
def format_transactions_metadata(record: dict, metadata: dict) -> dict:
    metadata["id"]= record.get("id")
    metadata["status"] = record.get("status")
    metadata["type"] = record.get("type")
    metadata["amount"] = record.get("amount")
    metadata["name"] = record.get("name")
    return metadata

def load_docs():
    loader = JSONLoader(
        file_path="../src/data/transactions.json",
        jq_schema=".[]",
        text_content=False,
        metadata_func=format_transactions_metadata
    )

    docs = loader.load()

    for doc in docs:
        data = json.loads(doc.page_content)
        transaction_direction = "Sent to" if data.get("type") == "debit" else "Recieved from"
        doc.page_content = (
            f"Transaction ID: {data.get('id')} | "
            f"{transaction_direction} {data.get('name')} ({data.get('upiId')}) | "
            f"Amount: ₹{data.get('amount')} | "
            f"Status: {data.get('status')} | "
            f"Method: {data.get('paymentMethod')} | "
            f"Date: {data.get('date')} | "
            f"Note: {data.get('note')} | "
            f"Bank Ref: {data.get('bankRef')}"
        )
    
    return docs

# chain
docs = load_docs()
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 30}
)

SYSTEM_PROMPT = """
You are PersonalPayBot, a personal transaction assistant for a PhonePe-style payments app.
Answer **ONLY** from the **CONTEXT** provided. Do not guess or hallucinate. Make sure to keep the answers short and clear.
Represent the long answer in points style. Use Rupees symbol for amounts. 
Keep answers short and clear. If you don't find any related answer, do not hallucinate, 
and simply tell that you can not find the transaction required by the user.
     
<context>
{context}
</context>
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human","{input}")
])

document_chain = create_stuff_documents_chain(llm=LLM, prompt=prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

# requests schema
class QueryRequest(BaseModel):
    query: str

# API Endpoint
@app.post("/chat")
def chat_api(request: QueryRequest):
    try:
        response = retriever_chain.invoke({"input": request.query})
        return {
            "status": "SUCCESS",
            "answer": response["answer"]
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e)
        }

# Health Checkup endpoint
@app.get("/")
def home():
    return {
        "message": "Transaction API is run"
    }