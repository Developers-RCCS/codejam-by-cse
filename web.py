import logging
from flask import Flask, render_template, request, jsonify, session
import os
from datetime import datetime
import json
import time
from agents.orchestrator import OrchestratorAgent
import traceback
from flask import send_file
import io
from gtts import gTTS

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

def list_user_chats(user_id):
    """List all chat files for a user, sorted by most recent."""
    chat_files = []
    for fname in os.listdir('chats'):
        if fname.startswith(user_id + '_') and fname.endswith('.json'):
            chat_files.append(fname)
    # Sort by timestamp descending
    chat_files.sort(reverse=True)
    return chat_files

def get_chat_title(chat_data, fallback):
    """Get a title for the chat, fallback to filename if not found."""
    # Try to get a title from the first user message
    for msg in chat_data:
        if msg.get('sender') == 'user' and msg.get('message'):
            return msg['message'][:30] + ('...' if len(msg['message']) > 30 else '')
    return fallback

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

@app.route('/api/chats', methods=['GET'])
def api_list_chats():
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    user_id = session['user_id']
    chat_files = list_user_chats(user_id)
    chats = []
    for fname in chat_files:
        path = os.path.join('chats', fname)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            title = get_chat_title(data, fname)
        except Exception:
            title = fname
        chats.append({'id': fname, 'title': title})
    return jsonify({'chats': chats})

@app.route('/api/chat', methods=['POST'])
def api_new_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    user_id = session['user_id']
    # Optionally accept a title, but we just create an empty chat
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_id = f"{user_id}_{timestamp}.json"
    path = os.path.join('chats', chat_id)
    with open(path, 'w') as f:
        json.dump([], f)
    return jsonify({'chat_id': chat_id})

@app.route('/api/chat/<chat_id>', methods=['GET'])
def api_get_chat(chat_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    user_id = session['user_id']
    if not chat_id.startswith(user_id + '_'):
        return jsonify({'error': 'Unauthorized'}), 403
    path = os.path.join('chats', chat_id)
    if not os.path.exists(path):
        return jsonify({'error': 'Chat not found'}), 404
    with open(path, 'r') as f:
        data = json.load(f)
    return jsonify({'chat': data})

@app.route('/api/chat/<chat_id>', methods=['DELETE'])
def api_delete_chat(chat_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    user_id = session['user_id']
    if not chat_id.startswith(user_id + '_'):
        return jsonify({'error': 'Unauthorized'}), 403
    path = os.path.join('chats', chat_id)
    if not os.path.exists(path):
        return jsonify({'error': 'Chat not found'}), 404
    os.remove(path)
    return jsonify({'success': True})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        # Return the audio file as response
        return send_file(mp3_fp, mimetype='audio/mpeg', as_attachment=False, download_name='speech.mp3')
    except Exception as e:
        logger.error(f"Error generating speech: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate speech'}), 500

if __name__ == '__main__':
    if orchestrator:
        logger.info("Starting Flask development server...")
        app.run(debug=True)
    else:
        logger.critical("Flask server cannot start because OrchestratorAgent failed to initialize.")