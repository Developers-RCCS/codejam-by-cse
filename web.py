from flask import Flask, render_template, request, jsonify, session
import faiss
import numpy as np
import pickle
from gemini_utils import embed_text, setup_gemini
import os
from datetime import datetime
import json
import sys
import glob
from gtts import gTTS
import logging;
import traceback;
import io;
from flask import send_file;

# Add debug print statements
print("Starting web application...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

app = Flask(__name__)
app.secret_key = 'dj89we923n7yr27y4x74y8x634txb6fx763t4x763tn47s6326st6s7t26nn73n6'

# Configuration
MAX_HISTORY_LENGTH = 20  # Maximum number of messages to keep in memory
CHATS_DIR = 'chats'
MAX_CONTEXT_LENGTH = 3000  # Max tokens for context window
SUMMARY_INTERVAL = 5       # Generate summary every 5 messages

# Load FAISS index + metadata
try:
    print("Attempting to load FAISS index...")
    index = faiss.read_index("faiss_index.index")
    print("FAISS index loaded successfully")

    print("Attempting to load metadata...")
    with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    print("Metadata loaded successfully")

    texts = metadata["texts"]
    metadatas = metadata["metadatas"]

    # Setup Gemini for answering
    print("Setting up Gemini...")
    gemini = setup_gemini()
    print("Gemini set up successfully")
except Exception as e:
    print(f"Error during initialization: {e}")
    # Continue anyway for debugging

# Ensure chats directory exists
if not os.path.exists(CHATS_DIR):
    os.makedirs(CHATS_DIR)
    print("Created chats directory")

def search_chunks(query, top_k=300):
    query_embedding = np.array(embed_text(query), dtype="float32").reshape(1, -1)
    D, I = index.search(query_embedding, top_k)

    results = []
    for idx in I[0]:
        results.append({
            "text": texts[idx],
            "page": metadatas[idx]["page"]
        })
    return results

def reasoning_agent(query, context_chunks, chat_history=None):
    context_text = "\n\n".join(
        [f"(Page {c['page']}) {c['text']}" for c in context_chunks]
    )

    # Construct conversation history
    conversation = ""
    if chat_history:
        trimmed = chat_history[-MAX_HISTORY_LENGTH:]  # Limit history length
        for msg in trimmed:
            role = "Student" if msg["sender"] == "user" else "Yuhasa"
            conversation += f"{role}: {msg['message']}\n"

    prompt = f"""
You are Yuhasa, a smart, calm, and kind female tutor helping a student understand history.

STRICT AND IMPORTANT: Use markdown styling in answers.

Always try to give the direct answer.
Do not always say "the text says...". Answer like you know the thing not like you have read from somewhere.

Always break long paragraphs into short readable ones.

You have access to several textbook excerpts. Your job is to:
1. Carefully read and interpret the context.
2. Piece together clues or references, even if the answer isn't directly stated.
3. Provide a thoughtful, reasoned answer — just like a human tutor would.
4. Stay consistent with what you've already said
5. Don't repeat answers unless helpful

✅ You are allowed to infer answers based on strong clues.
❌ You must not invent facts that contradict the context.
🧠 Think deeply and explain your reasoning if needed.

In case some relevant details are spread across multiple pages, try to combine them and infer the best possible answer using all the provided context.

---

Current date and time: {datetime.now().strftime('%Y-%m-%d')}

# IMPORTANT: If the textbook content provided does not have the relevent information, Use the chat history:
{conversation if chat_history else "No prior conversation history"}

📘 Textbook Context:
{context_text}

❓ Student Question:
{query}

💬 Your Answer (interpret and reason from the textbook + conversation):
"""
    return prompt

def generate_answer(query, context_chunks, chat_history=None):
    prompt = reasoning_agent(query, context_chunks, chat_history)
    response = gemini.generate_content(prompt)
    return response.text.strip()

def save_chat_history(user_id, chat_data):
    """Save complete chat session data including history and metadata"""
    os.makedirs(CHATS_DIR, exist_ok=True)
    filename = f"{CHATS_DIR}/{user_id}_chats.json"
    
    try:
        # Load existing chats if any
        existing_chats = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_chats = json.load(f)
        
        # Update or add the current chat session
        updated = False
        for i, chat in enumerate(existing_chats):
            if chat['id'] == chat_data['id']:
                existing_chats[i] = chat_data
                updated = True
                break
        
        if not updated:
            existing_chats.append(chat_data)
        
        # Save back to file
        with open(filename, 'w') as f:
            json.dump(existing_chats, f, indent=2)
            
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_chat_history(user_id):
    """Load all chat sessions for a user"""
    filename = f"{CHATS_DIR}/{user_id}_chats.json"
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat history: {e}")
    return []

@app.route('/')
def home():
    print("Received request for home page")
    # Initialize session if not already done
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()
        print(f"Created new user session: {session['user_id']}")
    else:
        print(f"Using existing session: {session['user_id']}")
    
    # Load chat history if any exists
    chat_history = load_chat_history(session['user_id'])
    print(f"Loaded chat history, entries: {len(chat_history)}")
    
    try:
        print("Attempting to render index.html template")
        return render_template('index.html', chat_history=chat_history)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return f"""
        <html>
        <head><title>Debug Page</title></head>
        <body>
        <h1>Template Error</h1>
        <p>There was an error rendering the template: {e}</p>
        <p>Current working directory: {os.getcwd()}</p>
        <p>Templates directory: {os.path.join(os.getcwd(), 'templates')}</p>
        </body>
        </html>
        """

def generate_summary(chat_history):
    """Generate a summary of the conversation so far"""
    summary_prompt = f"""
    Summarize this conversation briefly while preserving key details:
    {chat_history}
    """
    response = gemini.generate_content(summary_prompt)
    return response.text

@app.route('/ask', methods=['POST'])
def ask():
    if 'user_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    
    data = request.get_json()
    query = data.get('query', '')
    chat_history = data.get('chat_history', [])
    
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    # Generate summary periodically
    if len(chat_history) > 0 and len(chat_history) % SUMMARY_INTERVAL == 0:
        summary = generate_summary(chat_history)
        chat_history.append({'sender': 'system', 'message': f"Conversation summary: {summary}"})

    # Prepare context window (most recent messages first)
    context_messages = []
    token_count = 0
    for msg in reversed(chat_history):
        msg_content = f"{msg['sender']}: {msg['message']}"
        if token_count + len(msg_content.split()) > MAX_CONTEXT_LENGTH:
            break
        context_messages.insert(0, msg)  # Add to beginning to maintain order
        token_count += len(msg_content.split())

    # Process query with full context
    chunks = search_chunks(query)
    answer = generate_answer(query, chunks, context_messages)
    
    # Update history
    updated_history = chat_history.copy()
    updated_history.append({'sender': 'user', 'message': query})
    updated_history.append({'sender': 'bot', 'message': answer})
    
    # Save to disk
    save_chat_history(session['user_id'], updated_history)
    
    return jsonify({
        'answer': answer,
        'chat_history': updated_history
    })

logger = logging.getLogger(__name__)

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
    print("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0')