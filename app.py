from flask import Flask, render_template, request, session, redirect, url_for
from markdown import markdown
# Import the main orchestrator agent
from agents.orchestrator import OrchestratorAgent
import os # For secret key
import traceback # Import traceback for detailed error logging
import sys

# Add debug print statements
print("Starting application...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

app = Flask(__name__)
# Use an environment variable or generate a random key for production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_default_development_secret_key")

# List templates directory to verify files
templates_dir = os.path.join(os.getcwd(), 'templates')
if os.path.exists(templates_dir):
    print(f"Templates directory exists: {templates_dir}")
    print(f"Contents: {os.listdir(templates_dir)}")
else:
    print(f"Templates directory does not exist: {templates_dir}")

# Initialize the orchestrator (loads all sub-agents)
try:
    print("Initializing OrchestratorAgent...")
    orchestrator = OrchestratorAgent()
    print("OrchestratorAgent initialized successfully")
except Exception as e:
    print(f"❌ Error during orchestrator initialization: {e}")
    print(traceback.format_exc())
    # Create a mock orchestrator for debugging purposes
    class MockOrchestrator:
        def run(self, **kwargs):
            return {
                "answer": "The orchestrator failed to initialize. Please check the server logs.",
                "references": {"pages": [], "sections": []}
            }
    orchestrator = MockOrchestrator()
    print("Using MockOrchestrator due to initialization failure")

@app.route("/", methods=["GET", "POST"])
def chat():
    print("Received request for chat page")
    if "chat_history" not in session:
        session["chat_history"] = [] # Store dicts: {"user": ..., "bot": ..., "references": ...}
        print("Created new chat history in session")
    else:
        print(f"Using existing chat history, entries: {len(session['chat_history'])}")

    if request.method == "POST":
        user_query = request.form["query"]
        print(f"Processing POST request with query: '{user_query}'")
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
        try:
            print("Converting markdown answer to HTML")
            bot_answer_html = markdown(bot_answer_raw)
        except Exception as e:
            print(f"❌ Error converting markdown: {e}")
            bot_answer_html = f"<p>Error formatting response: {e}</p><pre>{bot_answer_raw}</pre>"

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
    try:
        print("Attempting to render chat.html template")
        return render_template("chat.html", chat_history=template_history)
    except Exception as e:
        print(f"❌ Error rendering template: {e}")
        print(traceback.format_exc())
        return f"""
        <html>
        <head><title>Template Error</title></head>
        <body>
        <h1>Template Error</h1>
        <p>There was an error rendering the template: {e}</p>
        <p>Current working directory: {os.getcwd()}</p>
        <p>Templates directory: {os.path.join(os.getcwd(), 'templates')}</p>
        <p>Template history entries: {len(template_history)}</p>
        </body>
        </html>
        """


@app.route("/clear")
def clear_chat():
    print("Clearing chat history")
    session.pop("chat_history", None)
    return redirect(url_for("chat"))

if __name__ == "__main__":
    print("Starting Flask development server...")
    # Set debug=False for production
    # Consider using a production server like Gunicorn or Waitress
    app.run(debug=True, host='0.0.0.0', port=5000) # Use port 5000 explicitly
