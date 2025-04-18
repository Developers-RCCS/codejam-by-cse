from flask import Flask, render_template, request, jsonify, session
import os
from datetime import datetime
import json
import time
from agents.orchestrator import OrchestratorAgent # Import Orchestrator

app = Flask(__name__)
app.secret_key = 'dj89we923n7yr27y4x74y8x634txb6fx763t4x763tn47s6326st6s7t26nn73n6'

# Ensure chats directory exists
if not os.path.exists('chats'):
    os.makedirs('chats')

# Instantiate Orchestrator Agent (globally)
# This assumes agents handle their own internal resource loading (FAISS, Gemini)
print("Initializing Orchestrator Agent for the web app...")
try:
    orchestrator = OrchestratorAgent()
    print("Orchestrator Agent initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize OrchestratorAgent: {e}")
    # In a real app, you might exit or prevent the app from starting
    orchestrator = None

def save_chat_history(user_id, messages):
    os.makedirs("chats", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chats/{user_id}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(messages, f)

def load_chat_history(user_id):
    # In a real app, you'd implement proper chat history loading
    # For now, we'll just return an empty list
    return []

@app.route('/')
def home():
    # Initialize session if not already done
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()
    
    # Load chat history if any exists
    chat_history = load_chat_history(session['user_id'])
    
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time() # Start total time measurement

    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401

    if orchestrator is None:
        return jsonify({'error': 'Chatbot service is unavailable due to initialization error.'}), 503

    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'Empty query'}), 400

    # Get chat history from request
    chat_history = data.get('chat_history', [])

    try:
        # --- Call Orchestrator --- 
        print(f"Web app received query: '{query}'. Calling orchestrator...")
        result = orchestrator.run(query=query, chat_history=chat_history)
        final_answer = result["answer"]
        # References could be accessed via result["references"] if needed for the UI later
        print(f"Orchestrator returned answer: {final_answer[:100]}...")
        # -------------------------

        # Update chat history with the FINAL answer from orchestrator
        chat_history.append({'sender': 'user', 'message': query})
        chat_history.append({'sender': 'bot', 'message': final_answer})

        # Save the updated chat history
        save_chat_history(session['user_id'], chat_history)

        total_time = time.time() - start_time # End total time measurement

        # --- Simplified Logging --- 
        print(f"Query: '{query}'")
        print(f"  Total Request Time (Web Route): {total_time:.4f}s")
        # -----------------------

        return jsonify({
            'answer': final_answer, # Return the final answer from orchestrator
            'chat_history': chat_history
        })

    except Exception as e:
        print(f"❌ Error during /ask route processing: {e}")
        return jsonify({'error': f'An internal error occurred: {e}'}), 500

if __name__ == '__main__':
    # Check if orchestrator initialized before running
    if orchestrator:
        print("Starting Flask development server...")
        app.run(debug=True)
    else:
        print("Flask server cannot start because OrchestratorAgent failed to initialize.")