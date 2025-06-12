import os
import PyPDF2
# or alternatively: import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return text

def process_pdfs_to_text(input_dir, output_dir):
    """Convert all PDFs in input directory to text files"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        
        # Save as text file
        output_file = output_path / f"{pdf_file.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    process_pdfs_to_text("input/pdfs", "input/processed")


