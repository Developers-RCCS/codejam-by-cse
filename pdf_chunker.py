"""Processes a PDF document, extracting text and splitting it into fixed-size, overlapping chunks with basic metadata."""
import fitz  # PyMuPDF
import re
import logging

logger = logging.getLogger(__name__)

def get_first_line_as_section(page_text: str) -> str:
    """Extracts the first non-empty line of text to use as a simple section title."""
    lines = page_text.strip().split('\n')
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            # Limit length and remove excessive whitespace
            return re.sub(r'\s+', ' ', cleaned_line[:100])
    return "Unknown Section" # Default if no text found

def chunk_text_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into chunks based on word count with overlap."""
    if not text:
        return []
    words = re.split(r'\s+', text.strip()) # Split by whitespace
    if not words:
        return []

    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = min(start_index + chunk_size, len(words))
        chunk = " ".join(words[start_index:end_index])
        chunks.append(chunk)
        # Move start index for the next chunk, considering overlap
        start_index += chunk_size - overlap
        if start_index >= len(words) - overlap: # Avoid tiny overlapping chunks at the end
             # If the remaining part is smaller than overlap, just break or handle differently?
             # For simplicity, let's just break to avoid very small last chunks.
             # A better approach might add the remainder if it's substantial.
             if end_index < len(words): # Add the very last bit if not captured
                 final_chunk = " ".join(words[start_index:])
                 if len(final_chunk.split()) > overlap // 2: # Only add if reasonably sized
                    chunks.append(final_chunk)
             break # Exit loop

    return chunks


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Loads a PDF, extracts text page by page, and creates fixed-size, overlapping word chunks.

    Each chunk includes metadata: the page number and the first line of text from that page
    as a simple section identifier.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): The target number of words per chunk.
        overlap (int): The number of words to overlap between consecutive chunks.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a chunk
                    containing 'text' and 'metadata' (page, section). Returns an
                    empty list if the PDF cannot be processed.
    """
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"📄 Successfully opened PDF: {pdf_path}, Pages: {doc.page_count}")
    except Exception as e:
        logger.error(f"❌ Failed to open PDF: {pdf_path}, Error: {e}", exc_info=True)
        return []

    all_chunks_with_metadata = []
    full_text = ""

    logger.info(f"Processing PDF '{pdf_path}' for chunking (Size: {chunk_size}, Overlap: {overlap})...")

    # Concatenate text from all pages first
    page_texts = {}
    for page_num, page in enumerate(doc):
        page_number = page_num + 1 # 1-based index
        page_text = page.get_text("text").strip()
        if page_text:
            page_texts[page_number] = page_text
            full_text += page_text + "\n\n" # Add separators between pages

    # Now chunk the entire text
    words = re.split(r'\s+', full_text.strip())
    current_chunk_start_word_index = 0

    while current_chunk_start_word_index < len(words):
        current_chunk_end_word_index = min(current_chunk_start_word_index + chunk_size, len(words))
        chunk_words = words[current_chunk_start_word_index:current_chunk_end_word_index]
        chunk_text = " ".join(chunk_words)

        # Determine the primary page number for this chunk (heuristic: page where the chunk starts)
        # This requires mapping word indices back to pages, which is complex.
        # Simpler approach: Assign page number based on where the *majority* of the chunk lies,
        # or just where it starts. Let's stick to the start for simplicity.

        # Find which page this chunk starts on (approximate)
        # This is still tricky with concatenated text. A simpler, page-by-page chunking is better.

        # --- Revision: Chunk page by page to keep metadata simple ---
        all_chunks_with_metadata = []
        processed_text_cache = "" # Keep track of text across page breaks for overlap

        for page_num, page in enumerate(doc):
            page_number = page_num + 1
            page_text = page.get_text("text").strip()
            if not page_text:
                continue

            page_section_title = get_first_line_as_section(page_text)
            logger.debug(f"Page {page_number}: Section='{page_section_title}'")

            # Combine remaining text from previous page (for overlap) with current page text
            text_to_chunk = processed_text_cache + page_text
            page_words = re.split(r'\s+', text_to_chunk.strip())

            page_chunk_start_index = 0
            while True:
                page_chunk_end_index = min(page_chunk_start_index + chunk_size, len(page_words))
                chunk_words = page_words[page_chunk_start_index:page_chunk_end_index]
                if not chunk_words:
                    break # No more words to process

                chunk_text = " ".join(chunk_words)
                all_chunks_with_metadata.append({
                    "text": chunk_text,
                    "metadata": {
                        "page": page_number,
                        "section": page_section_title
                    }
                })
                logger.debug(f"  Added chunk from page {page_number}, words {page_chunk_start_index}-{page_chunk_end_index}")

                # Prepare for next iteration
                next_start_index = page_chunk_start_index + chunk_size - overlap
                if next_start_index >= len(page_words) or page_chunk_end_index == len(page_words):
                     # If we reached the end of the page's words, store the overlap portion for the next page
                     overlap_start_index = max(0, len(page_words) - overlap)
                     processed_text_cache = " ".join(page_words[overlap_start_index:])
                     break # Move to the next page
                else:
                     page_chunk_start_index = next_start_index


    # --- End Revision ---


    logger.info(f"✅ PDF processed. Generated {len(all_chunks_with_metadata)} chunks using fixed-size word chunking.")
    doc.close() # Close the document
    return all_chunks_with_metadata

# Example usage (optional)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # pdf_file = "grade-11-history-text-book.pdf" # Make sure this file exists
    pdf_file = "non_existent_file.pdf" # Test error handling
    if pdf_file == "grade-11-history-text-book.pdf":
         pdf_chunks = load_and_chunk_pdf(pdf_file, chunk_size=400, overlap=100)
         logger.info(f"Generated {len(pdf_chunks)} chunks from {pdf_file}.")
         if pdf_chunks:
             logger.info("\n--- First Chunk Example ---")
             logger.info(f"Text ({len(pdf_chunks[0]['text'].split())} words): {pdf_chunks[0]['text'][:200]}...")
             logger.info(f"Metadata: {pdf_chunks[0]['metadata']}")
             logger.info("\n--- Last Chunk Example ---")
             logger.info(f"Text ({len(pdf_chunks[-1]['text'].split())} words): {pdf_chunks[-1]['text'][-200:]}...")
             logger.info(f"Metadata: {pdf_chunks[-1]['metadata']}")
    else:
         logger.warning(f"Skipping example usage for {pdf_file}. Set pdf_file to a valid path to run.")
