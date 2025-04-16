from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import pickle
from gemini_utils import embed_text, setup_gemini

# Initialize Gemini + FAISS
gemini = setup_gemini()
index = faiss.read_index("faiss_index.index")
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]
metadatas = metadata["metadatas"]

app = FastAPI()

# Optional: allow all CORS origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def search_chunks(query, top_k=5):
    query_emb = np.array(embed_text(query), dtype="float32").reshape(1, -1)
    D, I = index.search(query_emb, top_k)
    return [{"text": texts[i], "page": metadatas[i]["page"]} for i in I[0]]

def generate_answer(query, chunks):
    context = "\n\n".join([f"(Page {c['page']}) {c['text']}" for c in chunks])
    prompt = f"""
You are a helpful AI tutor helping a student understand history.
Use the following textbook excerpts to answer the question.

Textbook Context:
{context}

Question:
{query}

Answer:"""
    response = gemini.generate_content(prompt)
    return response.text.strip()

@app.post("/ask")
async def ask_bot(request: QueryRequest):
    query = request.query
    chunks = search_chunks(query)
    answer = generate_answer(query, chunks)
    pages = list(set([c["page"] for c in chunks]))

    return {
        "answer": answer,
        "pages": pages
    }
