from flask import Flask, render_template, request, session, redirect, url_for
from markdown import markdown
# Import agents instead of direct functions
from agents.retriever import RetrieverAgent
from agents.generator import GeneratorAgent
from agents.reference_tracker import ReferenceTrackerAgent

app = Flask(__name__)
app.secret_key = "ravi-bot-secret"  # required for session

# Initialize agents (load models/indexes)
retriever = RetrieverAgent()
generator = GeneratorAgent()
reference_tracker = ReferenceTrackerAgent()

@app.route("/", methods=["GET", "POST"])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_query = request.form["query"]

        # 1. Retrieve relevant chunks
        context_chunks = retriever.run(query=user_query)

        # 2. Generate answer
        bot_answer_raw = generator.run(query=user_query, context_chunks=context_chunks)

        # 3. Track references (optional for now, generator includes basic refs)
        references = reference_tracker.run(context_chunks=context_chunks)
        # You might integrate references more cleanly into the answer later

        # Convert markdown answer to HTML
        bot_answer_html = markdown(bot_answer_raw)

        # Store interaction in session
        session["chat_history"].append({"user": user_query, "bot": bot_answer_html, "references": references})
        session.modified = True # Important for mutable session data

        return redirect(url_for("chat"))

    return render_template("chat.html", chat_history=session["chat_history"])

@app.route("/clear")
def clear_chat():
    session.pop("chat_history", None)
    return redirect(url_for("chat"))

if __name__ == "__main__":
    # Use host='0.0.0.0' to make it accessible on the network if needed
    app.run(debug=True, host='0.0.0.0')
