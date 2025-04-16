import faiss
import numpy as np
import pickle
from gemini_utils import embed_text, setup_gemini

# Load FAISS index + metadata
index = faiss.read_index("faiss_index.index")

with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]
metadatas = metadata["metadatas"]

# Setup Gemini for answering
gemini = setup_gemini()

def search_chunks(query, top_k=5):
    query_embedding = np.array(embed_text(query), dtype="float32").reshape(1, -1)
    D, I = index.search(query_embedding, top_k)

    results = []
    for idx in I[0]:
        results.append({
            "text": texts[idx],
            "page": metadatas[idx]["page"]
        })
    return results

def generate_answer(query, context_chunks):
    context_text = "\n\n".join(
        [f"(Page {c['page']}) {c['text']}" for c in context_chunks]
    )

    prompt = f"""
Use the following textbook concepts to answer the question.

Textbook Context:
{context_text}

Question:
{query}

Answer:"""

    response = gemini.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    print("🧠 Bot is online. Ask a history question:")
    while True:
        query = input("\n🗣️ You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Bye bro!")
            break
        chunks = search_chunks(query)
        answer = generate_answer(query, chunks)
        print("\n🤖 Bot:\n", answer)
