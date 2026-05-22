from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Notes RAG Chatbot")
app.include_router(router)