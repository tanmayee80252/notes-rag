from fastapi import APIRouter
from pydantic import BaseModel
from app.chat import chat
from app.ingest import ingest_notes

router = APIRouter()

class Question(BaseModel):
    question: str

@router.post("/chat")
def ask(body: Question):
    return chat(body.question)

@router.post("/ingest")
def ingest():
    ingest_notes()
    return {"status": "Notes ingested successfully"}