import os
import pandas as pd
import PyPDF2
# or alternatively: import fitz  # PyMuPDF
from pathlib import Path
import json

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

def process_csv_to_text(csv_path, text_columns=None, combine_columns=True):
    """Convert CSV to text files"""
    try:
        df = pd.read_csv(csv_path)
        
        if text_columns is None:
            # Automatically detect text columns (non-numeric)
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if combine_columns:
            # Combine all text columns into single documents
            text_parts = []
            for col in text_columns:
                if col in df.columns:
                    text_parts.extend(df[col].dropna().astype(str).tolist())
            return "\n\n".join(text_parts)
        else:
            # Create separate documents for each row
            texts = []
            for _, row in df.iterrows():
                row_text = " ".join([f"{col}: {row[col]}" for col in text_columns if pd.notna(row[col])])
                texts.append(row_text)
            return texts
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def process_json_to_text(json_path, text_fields=None):
    """Convert JSON to text files"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]  # Convert single object to list
        
        texts = []
        for item in data:
            if text_fields:
                # Extract specific fields
                text_parts = [str(item.get(field, '')) for field in text_fields if item.get(field)]
                texts.append(" ".join(text_parts))
            else:
                # Convert entire object to text
                texts.append(json.dumps(item, indent=2))
        
        return texts
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

def copy_text_files(input_dir, output_dir):
    """Copy existing text files directly"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for txt_file in input_path.glob("*.txt"):
        print(f"Copying: {txt_file.name}")
        output_file = output_path / txt_file.name
        
        # Copy the file
        with open(txt_file, 'r', encoding='utf-8') as src:
            content = src.read()
        
        with open(output_file, 'w', encoding='utf-8') as dst:
            dst.write(content)
        
        print(f"Copied: {output_file}")

def process_all_inputs(input_dir, output_dir, csv_config=None, json_config=None):
    """Process all input files to text format"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    processed_files = 0
    
    # Process PDF files
    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing PDF: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        if text.strip():
            output_file = output_path / f"{pdf_file.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Saved: {output_file}")
            processed_files += 1
    
    # Copy existing text files
    for txt_file in input_path.glob("*.txt"):
        print(f"Copying text file: {txt_file.name}")
        output_file = output_path / txt_file.name
        with open(txt_file, 'r', encoding='utf-8') as src:
            content = src.read()
        with open(output_file, 'w', encoding='utf-8') as dst:
            dst.write(content)
        print(f"Copied: {output_file}")
        processed_files += 1
    
    # Process CSV files
    for csv_file in input_path.glob("*.csv"):
        print(f"Processing CSV: {csv_file.name}")
        
        # Use custom config or defaults
        config = csv_config or {}
        text_columns = config.get('text_columns', None)
        combine_columns = config.get('combine_columns', True)
        
        if combine_columns:
            text = process_csv_to_text(csv_file, text_columns, combine_columns=True)
            if text:
                output_file = output_path / f"{csv_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved: {output_file}")
                processed_files += 1
        else:
            texts = process_csv_to_text(csv_file, text_columns, combine_columns=False)
            if texts:
                for i, text in enumerate(texts):
                    output_file = output_path / f"{csv_file.stem}_row_{i+1}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    processed_files += 1
                print(f"Saved {len(texts)} text files from {csv_file.name}")
    
    # Process JSON files
    for json_file in input_path.glob("*.json"):
        print(f"Processing JSON: {json_file.name}")
        
        config = json_config or {}
        text_fields = config.get('text_fields', None)
        
        texts = process_json_to_text(json_file, text_fields)
        if texts:
            for i, text in enumerate(texts):
                output_file = output_path / f"{json_file.stem}_item_{i+1}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                processed_files += 1
            print(f"Saved {len(texts)} text files from {json_file.name}")
    
    print(f"\nTotal files processed: {processed_files}")
    return processed_files

if __name__ == "__main__":
    # Configuration for different file types
    csv_config = {
        'text_columns': None,  # Auto-detect or specify: ['column1', 'column2']
        'combine_columns': True  # True = one file, False = separate files per row
    }
    
    json_config = {
        'text_fields': None  # Auto-extract all or specify: ['title', 'content', 'description']
    }
    
    # Process all input files
    process_all_inputs("input/raw", "input/processed", csv_config, json_config)
