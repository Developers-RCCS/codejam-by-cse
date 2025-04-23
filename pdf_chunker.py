import fitz  # PyMuPDF
import re

def load_and_chunk_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into smaller chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append({
                "text": chunk,
                "page": page_num + 1  # page number for ref
            })
    return chunks
