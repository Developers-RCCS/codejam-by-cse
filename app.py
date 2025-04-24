from flask import Flask, render_template, request, session, redirect, url_for
import faiss
import numpy as np
import pickle
from gemini_utils import embed_text, setup_gemini
from markdown import markdown

app = Flask(__name__)
app.secret_key = "ravi-bot-secret"  # required for session

# Load FAISS + Gemini
faiss_index = faiss.read_index("faiss_index.index")
with open("faiss_metadata.pkl", "rb") as f:
    meta = pickle.load(f)

texts = meta["texts"]
metadatas = meta["metadatas"]
gemini = setup_gemini()

def search_chunks(query, top_k=5):
    emb = np.array(embed_text(query), dtype="float32").reshape(1, -1)
    D, I = faiss_index.search(emb, top_k)
    return [{"text": texts[i], "page": metadatas[i]["page"]} for i in I[0]]

def generate_answer(query, chunks):
    context = "\n\n".join([f"(Page {c['page']}) {c['text']}" for c in chunks])
    prompt = f"""
You are Histronaut, a helpful AI tutor helping a student understand history.
Use the following textbook excerpts to answer the question.

Textbook Context:
{context}

Question:
{query}

Answer:"""
    
    response = gemini.generate_content(prompt)
    return response.text.strip(), [c["page"] for c in chunks]

@app.route("/", methods=["GET", "POST"])
def chat():
    if "chat" not in session:
        session["chat"] = []

    if request.method == "POST":
        query = request.form["query"]
        session["chat"].append({"sender": "user", "text": query})

        try:
            chunks = search_chunks(query)
            answer, pages = generate_answer(query, chunks)
            session["chat"].append({
                "sender": "bot",
                "text": markdown(answer),  # Markdown to HTML here 👈
                "pages": pages
            })
        except Exception as e:
            session["chat"].append({
                "sender": "bot",
                "text": "❌ Yuhasa had a moment... error: " + str(e)
            })

        session.modified = True
        return redirect(url_for("chat"))

    return render_template("chat.html", chat=session["chat"])

@app.route("/clear")
def clear_chat():
    session.pop("chat", None)
    return redirect(url_for("chat"))

if __name__ == "__main__":
    app.run(debug=True)
