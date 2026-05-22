import os
from groq import Groq
from dotenv import load_dotenv
from app.retriever import retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(question):
    chunks, sources = retrieve(question)
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the notes provided below. If the answer isn't in the notes, say "I couldn't find that in your notes."

Notes:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources))
    }