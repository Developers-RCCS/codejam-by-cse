import logging
from flask import Flask, render_template, request, jsonify, session
import os
from datetime import datetime
import json
import time
from agents.retriever import RetrieverAgent
from agents.generator import GeneratorAgent
import traceback
from flask import send_file
import io

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

# Instantiate Retriever and Generator Agents (globally)
logger.info("Initializing RAG Agents for the web app...")
try:
    retriever = RetrieverAgent()
    generator = GeneratorAgent()
    logger.info("Retriever and Generator Agents initialized successfully.")
    agents_initialized = True
except Exception as e:
    logger.fatal(f"FATAL ERROR: Could not initialize RAG Agents: {e}", exc_info=True)
    retriever = None
    generator = None
    agents_initialized = False

def save_chat_to_file(chat_id, messages):
    """Saves the chat messages to a specific chat_id file."""
    if not chat_id:
        logger.error("Attempted to save chat with no chat_id.")
        return
    path = os.path.join('chats', chat_id)
    try:
        with open(path, 'w') as f:
            json.dump(messages, f, indent=2)
        logger.info(f"Chat history saved to {path}")
    except Exception as e:
        logger.error(f"Error saving chat history to {path}: {e}", exc_info=True)

def load_chat_from_file(chat_id):
    """Loads chat messages from a specific chat_id file."""
    if not chat_id:
        return []
    path = os.path.join('chats', chat_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chat history from {path}: {e}", exc_info=True)
        return []

def list_user_chats(user_id):
    """List all chat files for a user, sorted by most recent."""
    chat_files = []
    chats_dir = 'chats'
    if not os.path.exists(chats_dir):
        return []
    for fname in os.listdir(chats_dir):
        if fname.startswith(user_id + '_') and fname.endswith('.json'):
            chat_files.append(fname)
    chat_files.sort(key=lambda x: x.split('_')[-1].replace('.json', ''), reverse=True)
    return chat_files

def get_chat_title(chat_data, fallback):
    """Get a title for the chat, fallback to filename if not found."""
    for msg in chat_data:
        if msg.get('sender') == 'user' and msg.get('message'):
            title_text = msg['message']
            return title_text[:30] + ('...' if len(title_text) > 30 else '')
    return fallback

@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()
        logger.info(f"New session created with user_id: {session['user_id']}")
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()

    if 'user_id' not in session:
        logger.warning("Access denied to /ask: Session expired or invalid.")
        return jsonify({'error': 'Session expired'}), 401

    if not agents_initialized or retriever is None or generator is None:
        logger.error("RAG Agents not initialized. Cannot process request.")
        return jsonify({'error': 'Chatbot service is unavailable due to initialization error.'}), 503

    data = request.get_json()
    query = data.get('query', '')
    chat_id = data.get('chat_id')

    if not query:
        logger.warning("Received empty query.")
        return jsonify({'error': 'Empty query'}), 400

    if not chat_id:
        logger.warning("Received request without chat_id.")
        return jsonify({'error': 'Missing chat ID'}), 400

    chat_history = load_chat_from_file(chat_id)

    try:
        logger.info(f"Processing query for chat_id {chat_id}: '{query}'")

        retrieval_start_time = time.time()
        logger.info("Step 1: Retrieving context...")
        retrieved_chunks = retriever.run(query=query)
        logger.info(f"Step 1: Retrieval complete ({len(retrieved_chunks)} chunks). Duration: {time.time() - retrieval_start_time:.4f}s")

        if retrieved_chunks:
            log_chunks = []
            for i, chunk in enumerate(retrieved_chunks):
                meta = chunk.get('metadata', {})
                log_chunks.append(f"  Chunk {i+1}: p.{meta.get('page', '?')}, sec: {meta.get('section', 'Unknown Section')}, score: {chunk.get('score', -1):.4f}")
            logger.info("Retrieved Chunk Metadata:\n" + "\n".join(log_chunks))
        else:
            logger.info("No chunks retrieved.")

        generation_start_time = time.time()
        logger.info("Step 2: Generating answer...")
        final_answer = generator.run(query=query, context_chunks=retrieved_chunks)
        logger.info(f"Step 2: Generation complete. Duration: {time.time() - generation_start_time:.4f}s")
        logger.info(f"Final Answer generated: {final_answer[:150]}...")

        chat_history.append({'sender': 'user', 'message': query})
        chat_history.append({'sender': 'bot', 'message': final_answer})

        save_chat_to_file(chat_id, chat_history)

        total_time = time.time() - start_time
        logger.info(f"Query processed successfully. Total time: {total_time:.4f}s")

        return jsonify({'answer': final_answer})

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Error during /ask route processing for chat_id {chat_id}: {e}", exc_info=True)
        logger.error(f"  Request processing time before error: {total_time:.4f}s")
        return jsonify({'error': f'An internal error occurred: {e}' }), 500

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
        except Exception as e:
            logger.warning(f"Could not load or parse chat file {fname} for title: {e}")
            title = fname
        chats.append({'id': fname, 'title': title})
    return jsonify({'chats': chats})

@app.route('/api/chat', methods=['POST'])
def api_new_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    user_id = session['user_id']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    chat_id = f"{user_id}_{timestamp}.json"
    path = os.path.join('chats', chat_id)
    try:
        with open(path, 'w') as f:
            json.dump([], f)
        logger.info(f"Created new chat file: {path}")
        return jsonify({'chat_id': chat_id})
    except Exception as e:
        logger.error(f"Failed to create new chat file {path}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to create new chat'}), 500

@app.route('/api/chat/<chat_id>', methods=['GET'])
def api_get_chat(chat_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    user_id = session['user_id']
    if not chat_id.startswith(user_id + '_'):
        logger.warning(f"Unauthorized attempt to access chat {chat_id} by user {user_id}")
        return jsonify({'error': 'Unauthorized'}), 403
    
    chat_data = load_chat_from_file(chat_id)
    if chat_data is None:
         return jsonify({'error': 'Chat not found or failed to load'}), 404
    
    return jsonify({'chat': chat_data})

@app.route('/api/chat/<chat_id>', methods=['DELETE'])
def api_delete_chat(chat_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    user_id = session['user_id']
    if not chat_id.startswith(user_id + '_'):
        logger.warning(f"Unauthorized attempt to delete chat {chat_id} by user {user_id}")
        return jsonify({'error': 'Unauthorized'}), 403
    path = os.path.join('chats', chat_id)
    if not os.path.exists(path):
        return jsonify({'error': 'Chat not found'}), 404
    try:
        os.remove(path)
        logger.info(f"Deleted chat file: {path}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to delete chat file {path}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete chat'}), 500

if __name__ == '__main__':
    if agents_initialized:
        logger.info("Starting Flask development server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.critical("Flask server cannot start because RAG Agents failed to initialize.")