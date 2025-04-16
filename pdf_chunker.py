import fitz  # PyMuPDF
import re

def extract_sections(page):
    """Very basic heuristic to find potential section headers."""
    # Look for lines in bold or larger font size (requires more advanced analysis)
    # Simple approach: Look for short lines, all caps, or ending with numbers
    # This is highly dependent on PDF structure and likely needs refinement.
    blocks = page.get_text("dict")["blocks"]
    headers = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                line_text = "".join([span["text"] for span in line["spans"]]).strip()
                # Heuristic: Short line, likely bold (check font flags if needed), maybe all caps
                is_short = len(line_text.split()) < 7
                is_all_caps = line_text.isupper() and len(line_text) > 3
                # Add more heuristics based on textbook structure
                if line_text and (is_short or is_all_caps):
                     # Basic check for font boldness (can be unreliable)
                     is_bold = any(span["flags"] & 16 for span in line["spans"])
                     if is_bold or is_all_caps:
                         headers.append(line_text)
    return headers[-1] if headers else "Unknown Section"  # Return last potential header

def load_and_chunk_pdf(pdf_path, chunk_size=500, overlap=50):
    """Loads PDF, chunks text respecting paragraphs, adds metadata."""
    doc = fitz.open(pdf_path)
    chunks = []
    current_chunk_text = ""
    current_section = "Unknown Section"
    para_count_in_chunk = 0

    print(f"📄 Processing PDF: {pdf_path}...")

    for page_num, page in enumerate(doc):
        page_number = page_num + 1  # 1-based page number
        # current_section = extract_sections(page)  # Basic section detection
        text_blocks = page.get_text("blocks")  # Get text blocks (often correspond to paragraphs)

        for i, block in enumerate(text_blocks):
            block_text = ""
            if block[4]:  # Check if block contains text
                block_text = block[4].strip()

            if not block_text:
                continue

            # Simple paragraph splitting (split by double newlines or treat each block as paragraph)
            paragraphs = re.split(r'\n\s*\n', block_text)  # Split block by double newlines

            for para_index, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # Attempt to detect section headers within blocks (if not done per page)
                # Heuristic: Short paragraph, maybe all caps or ends with number
                if len(paragraph.split()) < 7 and (paragraph.isupper() or paragraph[-1].isdigit()):
                    current_section = paragraph
                    # Don't add headers as part of chunk text? Optional.
                    # continue

                para_len = len(paragraph)
                if len(current_chunk_text) + para_len <= chunk_size:
                    current_chunk_text += (" " if current_chunk_text else "") + paragraph
                    para_count_in_chunk += 1
                else:
                    # Chunk is full, save it
                    if current_chunk_text:
                        chunks.append({
                            "text": current_chunk_text,
                            "metadata": {
                                "page": page_number,
                                "section": current_section,
                                "paragraphs": para_count_in_chunk
                            }
                        })
                    # Start new chunk with overlap (optional, simple overlap here)
                    # A better overlap would take previous words/sentences
                    overlap_text = current_chunk_text[-overlap:] if overlap > 0 else ""
                    current_chunk_text = overlap_text + (" " if overlap_text else "") + paragraph
                    para_count_in_chunk = 1

    # Add the last remaining chunk
    if current_chunk_text:
        chunks.append({
            "text": current_chunk_text,
            "metadata": {
                "page": page_number,  # Page number of the last processed page
                "section": current_section,
                "paragraphs": para_count_in_chunk
            }
        })

    print(f"✅ PDF processed. Generated {len(chunks)} chunks.")
    return chunks

# Example usage (optional)
# if __name__ == '__main__':
#     pdf_chunks = load_and_chunk_pdf("grade-11-history-text-book.pdf")
#     print(f"Generated {len(pdf_chunks)} chunks.")
#     if pdf_chunks:
#         print("\n--- First Chunk Example ---")
#         print(pdf_chunks[0]["text"])
#         print("Metadata:", pdf_chunks[0]["metadata"])
#         print("\n--- Last Chunk Example ---")
#         print(pdf_chunks[-1]["text"])
#         print("Metadata:", pdf_chunks[-1]["metadata"])
