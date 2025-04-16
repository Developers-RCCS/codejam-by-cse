from flask import Flask, render_template, request, session, redirect, url_for
from markdown import markdown
# Import the main orchestrator agent
from agents.orchestrator import OrchestratorAgent
import os # For secret key

app = Flask(__name__)
# Use an environment variable or generate a random key for production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_default_development_secret_key")

# Initialize the orchestrator (loads all sub-agents)
orchestrator = OrchestratorAgent()

@app.route("/", methods=["GET", "POST"])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = [] # Store dicts: {"user": ..., "bot": ..., "references": ...}

    if request.method == "POST":
        user_query = request.form["query"]

        # Run the orchestrator
        # Pass Flask session history if needed (adapt format if necessary)
        # For now, passing empty history to orchestrator, Flask manages its own
        result = orchestrator.run(query=user_query, chat_history=[]) # Pass empty for now

        # Convert markdown answer to HTML
        bot_answer_html = markdown(result["answer"])

        # Store interaction in session
        session["chat_history"].append({
            "user": user_query,
            "bot": bot_answer_html,
            "references": result["references"] # Store structured references
        })
        session.modified = True # Important!

        return redirect(url_for("chat"))

    # Pass history in the correct format for the template
    template_history = session.get("chat_history", [])
    return render_template("chat.html", chat_history=template_history)


@app.route("/clear")
def clear_chat():
    session.pop("chat_history", None)
    return redirect(url_for("chat"))

if __name__ == "__main__":
    # Set debug=False for production
    # Consider using a production server like Gunicorn or Waitress
    app.run(debug=True, host='0.0.0.0', port=5000) # Use port 5000 explicitly
