from flask import Flask, render_template, request, session, redirect, url_for
from markdown import markdown
# Import the main orchestrator agent
from agents.orchestrator import OrchestratorAgent
import os # For secret key
import traceback # Import traceback for detailed error logging

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
        bot_answer_raw = "Sorry, something went wrong." # Default error message
        references = {"pages": [], "sections": []} # Default empty references

        try:
            # Run the orchestrator
            print(f"Running orchestrator for query: {user_query}") # Log query
            result = orchestrator.run(query=user_query, chat_history=[]) # Pass empty for now
            bot_answer_raw = result["answer"]
            references = result["references"]
            print("Orchestrator run successful.") # Log success
        except Exception as e:
            # Log the full error traceback to the terminal
            print(f"❌ Error during orchestrator run for query '{user_query}':")
            print(traceback.format_exc())
            # Update the raw answer to show an error message to the user
            bot_answer_raw = f"Sorry, I encountered an error processing your request. Please check the server logs for details.\n\n*Error: {e}*"

        # Convert markdown answer to HTML (always happens, even on error)
        bot_answer_html = markdown(bot_answer_raw)

        # Store interaction in session (always happens)
        session["chat_history"].append({
            "user": user_query,
            "bot": bot_answer_html,
            "references": references # Store structured references (might be empty on error)
        })
        session.modified = True # Important!
        print("Session updated.") # Log session update

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
