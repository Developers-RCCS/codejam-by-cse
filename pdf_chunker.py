import fitz  # PyMuPDF
import re
from collections import defaultdict

def detect_sections(doc):
    """Attempts to detect section headers throughout the document based on formatting clues."""
    section_markers = defaultdict(list)  # Maps page numbers to detected section headers
    
    for page_num, page in enumerate(doc):
        # Get page structure including text blocks and their formatting
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    spans = line["spans"]
                    if not spans:
                        continue
                        
                    line_text = "".join([span["text"] for span in line["spans"]]).strip()
                    if not line_text:
                        continue
                    
                    # Heuristics for identifying section headers
                    # 1. Text is short (likely a title)
                    is_short = len(line_text.split()) < 6
                    
                    # 2. Text uses larger font than average
                    font_sizes = [span["size"] for span in spans]
                    avg_size = sum(font_sizes) / len(font_sizes)
                    is_larger_font = any(size > 12 for size in font_sizes)
                    
                    # 3. Text is bold or all caps
                    is_bold = any(span["flags"] & 16 for span in spans)
                    is_all_caps = line_text.isupper() and len(line_text) > 3
                    
                    # 4. Has numeric prefix like "1.2" or "Chapter V"
                    has_numeric_prefix = bool(re.match(r'^(\d+\.|\d+\.\d+|Chapter \w+)', line_text))
                    
                    # If any two conditions are true, consider it a section
                    if sum([is_short, is_larger_font, is_bold, is_all_caps, has_numeric_prefix]) >= 2:
                        section_markers[page_num+1].append(line_text)
    
    return section_markers

def load_and_chunk_pdf(pdf_path, chunk_size=500, overlap=100):
    """Loads PDF, chunks text respecting paragraphs, adds metadata with section detection."""
    doc = fitz.open(pdf_path)
    chunks = []
    
    # First pass: detect sections across the document
    section_markers = detect_sections(doc)
    
    # Second pass: extract content with section awareness
    current_chunk_text = ""
    current_section = "Introduction"  # Default section
    para_count_in_chunk = 0
    current_page = 1
    
    print(f"📄 Processing PDF: {pdf_path}...")
    
    for page_num, page in enumerate(doc):
        page_number = page_num + 1  # 1-based page number
        
        # Update current section if we have markers for this page
        if page_number in section_markers and section_markers[page_number]:
            current_section = section_markers[page_number][0]  # Use first detected section on page
        
        text_blocks = page.get_text("blocks")  # Get text blocks
        
        for i, block in enumerate(text_blocks):
            if not block[4]:  # Skip empty blocks
                continue
                
            block_text = block[4].strip()
            if not block_text:
                continue
            
            # Check if this block might be a section header
            if page_number in section_markers and block_text in section_markers[page_number]:
                current_section = block_text
                continue  # Skip adding the section header itself to chunks
            
            # Split block into paragraphs
            paragraphs = re.split(r'\n\s*\n', block_text)
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If adding this paragraph would exceed chunk size, save current chunk
                if len(current_chunk_text) + len(paragraph) > chunk_size and current_chunk_text:
                    chunks.append({
                        "text": current_chunk_text,
                        "metadata": {
                            "page": current_page,
                            "section": current_section,
                            "paragraphs": para_count_in_chunk
                        }
                    })
                    
                    # Start new chunk with overlap
                    words = current_chunk_text.split()
                    overlap_word_count = min(len(words), overlap // 5)  # ~5 chars per word average
                    overlap_text = " ".join(words[-overlap_word_count:]) if overlap_word_count > 0 else ""
                    current_chunk_text = overlap_text + " " + paragraph if overlap_text else paragraph
                    para_count_in_chunk = 1
                else:
                    # Add to current chunk
                    current_chunk_text += (" " if current_chunk_text else "") + paragraph
                    para_count_in_chunk += 1
                
                current_page = page_number  # Update current page
    
    # Add the final chunk
    if current_chunk_text:
        chunks.append({
            "text": current_chunk_text,
            "metadata": {
                "page": current_page,
                "section": current_section,
                "paragraphs": para_count_in_chunk
            }
        })
    
    print(f"✅ PDF processed. Generated {len(chunks)} chunks with section detection.")
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
