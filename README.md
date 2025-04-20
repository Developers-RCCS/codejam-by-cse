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
6. Ensure your `GOOGLE_API_KEY` is set in a `.env` file in the project root.

## Usage
Run the main script from the project root directory to process the PDF and create the vector store:
```bash
python src/main.py
```
