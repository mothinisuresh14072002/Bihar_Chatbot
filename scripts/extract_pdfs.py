"""
Extract text from BOCW PDF documents with structure preservation.
Creates clean, section-aware text files with proper paragraph boundaries.
"""
import sys
import re
import json
from pathlib import Path
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PDF_FILES, RAW_DATA_DIR, DATA_DIR


def is_heading(text, fontsize=0):
    """Heuristic: detect if a line is a section heading."""
    text = text.strip()
    if not text:
        return False
    # All-caps short lines are usually headings
    if text.isupper() and len(text) < 100:
        return True
    # Lines starting with numbers like "1.", "1.1", "Module 1:" etc.
    if re.match(r'^\d+(\.\d+)*[\.\)\:]?\s+', text) and len(text) < 120:
        return True
    # Lines starting with "Chapter", "Module", "Section"
    if re.match(r'^(chapter|module|section|appendix|table\s+of)\s', text, re.I):
        return True
    return False


def clean_line(text):
    """Clean a single line of extracted text."""
    # Remove excessive spaces (but keep single spaces)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove page number patterns like "Page 1 of 50"
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.I)
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text)
    return text.strip()


def extract_pdf_structured(pdf_path):
    """
    Extract text from PDF preserving structure.
    Returns list of sections: [{heading, content, page}]
    """
    print(f"  📄 Opening: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    
    sections = []
    current_heading = "Introduction"
    current_content = []
    current_page = 1
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Get text with blocks to preserve structure
        blocks = page.get_text("blocks")  # Returns list of (x0,y0,x1,y1,text,block_no,block_type)
        
        for block in blocks:
            if block[6] != 0:  # Skip image blocks
                continue
            
            raw_text = block[4]
            lines = raw_text.split('\n')
            
            for line in lines:
                cleaned = clean_line(line)
                if not cleaned:
                    continue
                
                if is_heading(cleaned):
                    # Save previous section if it has content
                    if current_content:
                        full_text = ' '.join(current_content)
                        if len(full_text.strip()) > 30:  # Skip very short sections
                            sections.append({
                                "heading": current_heading,
                                "content": full_text.strip(),
                                "page": current_page,
                                "source": pdf_path.name
                            })
                    
                    current_heading = cleaned
                    current_content = []
                    current_page = page_num + 1
                else:
                    current_content.append(cleaned)
    
    # Don't forget the last section
    if current_content:
        full_text = ' '.join(current_content)
        if len(full_text.strip()) > 30:
            sections.append({
                "heading": current_heading,
                "content": full_text.strip(),
                "page": current_page,
                "source": pdf_path.name
            })
    
    doc.close()
    return sections


def write_structured_output(sections, output_txt_path, output_json_path):
    """Write both plain text (for indexing) and JSON (with metadata)."""
    
    # Plain text: each section separated by clear markers
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for sec in sections:
            f.write(f"\n\n=== {sec['heading']} ===\n")
            f.write(sec['content'])
            f.write("\n")
    
    # JSON: structured data with metadata
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("  BOCW PDF Extractor — Structured Extraction")
    print("=" * 60)
    
    pdf_output_map = {
        "SRS_BOCW v2.4.pdf": ("pdf_srs_bocw", "SRS Document"),
        "User Manual on State BOCW DLC.pdf": ("pdf_user_manual", "User Manual"),
    }
    
    total_sections = 0
    
    for pdf_path in PDF_FILES:
        if not pdf_path.exists():
            print(f"  ❌ PDF not found: {pdf_path}")
            continue
        
        base_name, doc_type = pdf_output_map.get(
            pdf_path.name, (pdf_path.stem, "Document")
        )
        
        txt_path = RAW_DATA_DIR / f"{base_name}.txt"
        json_path = RAW_DATA_DIR / f"{base_name}.json"
        
        print(f"\n  📖 Extracting: {pdf_path.name} ({doc_type})")
        sections = extract_pdf_structured(pdf_path)
        
        write_structured_output(sections, txt_path, json_path)
        
        total_sections += len(sections)
        print(f"  ✅ Extracted {len(sections)} sections → {txt_path.name}")
        print(f"     Metadata saved → {json_path.name}")
        
        # Print section summary
        for i, sec in enumerate(sections[:5]):
            preview = sec['content'][:80] + "..." if len(sec['content']) > 80 else sec['content']
            print(f"     [{i+1}] {sec['heading']}: {preview}")
        if len(sections) > 5:
            print(f"     ... and {len(sections)-5} more sections")
    
    print(f"\n{'=' * 60}")
    print(f"  Total sections extracted: {total_sections}")
    print(f"  Output directory: {RAW_DATA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
