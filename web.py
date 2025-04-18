import logging
from flask import Flask, render_template, request, jsonify, session
import os
from datetime import datetime
import json
import time
from agents.orchestrator import OrchestratorAgent
import traceback

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

app = Flask(__name__)
app.secret_key = 'dj89we923n7yr27y4x74y8x634txb6fx763t4x763tn47s6326st6s7t26nn73n6'

# Ensure chats directory exists
if not os.path.exists('chats'):
    os.makedirs('chats')

# Instantiate Orchestrator Agent (globally)
logger.info("Initializing Orchestrator Agent for the web app...")
try:
    orchestrator = OrchestratorAgent()
    logger.info("Orchestrator Agent initialized successfully.")
except Exception as e:
    logger.fatal(f"FATAL ERROR: Could not initialize OrchestratorAgent: {e}", exc_info=True)
    orchestrator = None

def save_chat_history(user_id, messages):
    os.makedirs("chats", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chats/{user_id}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(messages, f)

def load_chat_history(user_id):
    return []

@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()
    
    chat_history = load_chat_history(session['user_id'])
    
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()

    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401

    if orchestrator is None:
        logger.error("Orchestrator not initialized. Cannot process request.")
        return jsonify({'error': 'Chatbot service is unavailable due to initialization error.'}), 503

    data = request.get_json()
    query = data.get('query', '')

    if not query:
        logger.warning("Received empty query.")
        return jsonify({'error': 'Empty query'}), 400

    chat_history = data.get('chat_history', [])

    try:
        logger.info(f"Web app received query: '{query}'. Calling orchestrator...")
        result = orchestrator.run(query=query, chat_history=chat_history)
        final_answer = result["answer"]
        logger.info(f"Orchestrator returned answer: {final_answer[:100]}...")

        chat_history.append({'sender': 'user', 'message': query})
        chat_history.append({'sender': 'bot', 'message': final_answer})

        save_chat_history(session['user_id'], chat_history)

        total_time = time.time() - start_time
        logger.debug(f"Query: '{query}'")
        logger.debug(f"  Total Request Time (Web Route): {total_time:.4f}s")

        return jsonify({
            'answer': final_answer,
            'chat_history': chat_history
        })

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Error during /ask route processing: {e}", exc_info=True)
        logger.error(f"  Request processing time before error: {total_time:.4f}s")
        return jsonify({'error': f'An internal error occurred: {e}'}), 500

if __name__ == '__main__':
    if orchestrator:
        logger.info("Starting Flask development server...")
        app.run(debug=True)
    else:
        logger.critical("Flask server cannot start because OrchestratorAgent failed to initialize.")