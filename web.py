from flask import Flask, render_template, request, jsonify, session
import faiss
import numpy as np
import pickle
from gemini_utils import embed_text, setup_gemini
import os
from datetime import datetime
import json
import time
import functools
import google.generativeai as genai # Ensure this is imported
import re
from agents.retriever import RetrieverAgent
from agents.query_analyzer import QueryAnalyzerAgent # Added import

app = Flask(__name__)
app.secret_key = 'dj89we923n7yr27y4x74y8x634txb6fx763t4x763tn47s6326st6s7t26nn73n6'

# Load FAISS index + metadata
index = faiss.read_index("faiss_index.index")

with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]
metadatas = metadata["metadatas"]

# Setup Gemini for answering
gemini = setup_gemini()

# Ensure chats directory exists
if not os.path.exists('chats'):
    os.makedirs('chats')

# Instantiate Agents
retriever_agent = RetrieverAgent()
query_analyzer_agent = QueryAnalyzerAgent() # Instantiate Query Analyzer

# --- Hybrid search function ---
# This function now primarily orchestrates the agents
def hybrid_search_chunks(query, top_k=5, initial_top_k=15):
    # 1. Analyze the query using the dedicated agent
    query_analysis = query_analyzer_agent.run(query) # Use the agent

    # 2. Retrieve using the analysis results
    # Use RetrieverAgent's hybrid run method, passing the analysis
    results = retriever_agent.run(query=query, query_analysis=query_analysis, initial_top_k=initial_top_k, final_top_k=top_k)
    
    # Format for downstream (page in metadata)
    return [{"text": r["text"], "page": r["metadata"].get("page", "?")} for r in results]

def reasoning_agent(query, context_chunks, chat_history=None):
    context_text = "\n\n".join(
        [f"(Page {c['page']}) {c['text']}" for c in context_chunks]
    )

    # 🧠 Construct real-time dialogue memory
    conversation = ""
    if chat_history:
        trimmed = chat_history[-50:]
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
5. Don’t repeat answers unless helpful

✅ You are allowed to infer answers based on strong clues.
❌ You must not invent facts that contradict the context.
🧠 Think deeply and explain your reasoning if needed.

In case some relevant details are spread across multiple pages, try to combine them and infer the best possible answer using all the provided context.

---

🧠 Chat History:
{conversation}

📘 Textbook Context:
{context_text}

❓ Student Question:
{query}

💬 Your Answer (interpret and reason from the textbook + conversation):
"""
    return prompt


def generate_answer(query, context_chunks, chat_history=None):
    prompt = reasoning_agent(query, context_chunks, chat_history)
    
    # --- Add Generation Config ---
    generation_config = genai.types.GenerationConfig(
        # candidate_count=1, # Default is 1
        # stop_sequences=None,
        # max_output_tokens=2048, # Adjust if needed, but Flash default is generous (8192)
        temperature=0.2, # Lower temperature for more focused, potentially faster responses
        # top_p=None,
        # top_k=None,
    )
    # ---------------------------

    response = gemini.generate_content(
        prompt,
        generation_config=generation_config # Pass the config here
    )
    return response.text.strip()

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
    
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    
    # Get chat history from request or session
    chat_history = data.get('chat_history', [])
    
    # --- Profiling Retrieval ---
    retrieval_start = time.time()
    # Use hybrid search (which now uses QueryAnalyzer and Retriever agents)
    chunks = hybrid_search_chunks(query, top_k=5, initial_top_k=15) # This call remains the same externally
    retrieval_time = time.time() - retrieval_start
    # --------------------------

    # --- Profiling Generation ---
    generation_start = time.time()
    answer = generate_answer(query, chunks, chat_history) # Pass chat_history here
    generation_time = time.time() - generation_start
    # --------------------------
    
    # Update chat history
    chat_history.append({'sender': 'user', 'message': query})
    chat_history.append({'sender': 'bot', 'message': answer})
    
    # Save the updated chat history
    save_chat_history(session['user_id'], chat_history)

    total_time = time.time() - start_time # End total time measurement

    # --- Logging Timings ---
    print(f"Query: '{query}'")
    print(f"  Retrieval Time (top_k={len(chunks)}): {retrieval_time:.4f}s")
    print(f"  Generation Time: {generation_time:.4f}s")
    print(f"  Total Request Time: {total_time:.4f}s")
    # -----------------------
    
    return jsonify({
        'answer': answer,
        'chat_history': chat_history
    })

if __name__ == '__main__':
    app.run(debug=True)