import sys
import re
from pathlib import Path
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PDF_FILES, RAW_DATA_DIR

def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Basic encoding fix/replace if needed, though PyMuPDF usually handles unicode well
    text = text.strip()
    return text

def extract_pdf_to_text(pdf_path, output_path):
    print(f"Extracting {pdf_path.name}...")
    try:
        doc = fitz.open(pdf_path)
        all_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text:
                all_text.append(clean_text(text))
                
        full_text = "\n".join(all_text)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"Extracted {len(doc)} pages to {output_path.name}")
    except Exception as e:
        print(f"Error extracting {pdf_path.name}: {e}")

def main():
    # Map the expected output names
    pdf_output_map = {
        "SRS_BOCW v2.4.pdf": "pdf_srs_bocw.txt",
        "User Manual on State BOCW DLC.pdf": "pdf_user_manual.txt"
    }
    
    for pdf_path in PDF_FILES:
        if pdf_path.exists():
            out_name = pdf_output_map.get(pdf_path.name, f"{pdf_path.stem}.txt")
            output_path = RAW_DATA_DIR / out_name
            extract_pdf_to_text(pdf_path, output_path)
        else:
            print(f"PDF not found: {pdf_path}")
            
if __name__ == "__main__":
    main()
