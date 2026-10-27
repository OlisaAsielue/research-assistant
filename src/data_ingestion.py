# src/data_ingestion.py
import fitz  # PyMuPDF
from pathlib import Path
import re

def clean_text(text: str) -> str:
    """Basic text cleaning for research papers"""
    text = re.sub(r'\[\d+\]', '', text)  # Remove citations [1]
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    return text.strip()

def process_pdfs(input_dir: str = "data/raw_pdfs", output_dir: str = "data/processed"):
    """Process all PDFs in input directory"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for pdf_path in Path(input_dir).glob("*.pdf"):
        with fitz.open(pdf_path) as doc:
            full_text = []
            for page in doc:
                text = page.get_text()
                full_text.append(clean_text(text))
            
            # Save as TXT
            output_path = Path(output_dir) / f"{pdf_path.stem}.txt"
            output_path.write_text("\n".join(full_text))
            print(f"Processed: {pdf_path.name} -> {output_path}")

if __name__ == "__main__":
    process_pdfs()
