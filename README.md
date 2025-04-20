# History RAG Chatbot (Future Minds Competition)

This project aims to build a multi-agent RAG chatbot using Gemini 1.5 Flash for the Future Minds competition.

## Current Status
Project initialized. PDF processing and vector store creation implemented.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Ensure the `history_textbook.pdf` file is located in the `data/raw/` directory. (Updated path)
6. Create a `.env` file in the project root directory.
7. Add your Google API key to the `.env` file like this:
   ```
   GOOGLE_API_KEY="YOUR_API_KEY_HERE"
   ```
   *(Ensure `.env` is added to your `.gitignore` if using Git)*

## Usage
Run the main script from the project root directory. It will first check if the vector store exists. If not, it will process the PDF and create it. Then, it will run a sample query using the RAG pipeline:
```bash
python src/main.py
```
