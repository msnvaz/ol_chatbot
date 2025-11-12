import pdfplumber
import os
import sys

def extract_text_from_pdfs(folder):
    """Extract text from all PDF files in the specified folder."""
    if not os.path.exists(folder):
        print(f"❌ Error: Folder '{folder}' does not exist!")
        print(f"Please create it and add your PDF textbooks.")
        sys.exit(1)
    
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{folder}'!")
        print(f"Please add your O/L textbook PDFs to this folder.")
        sys.exit(1)
    
    all_text = ""
    successful_extractions = 0
    
    for file in pdf_files:
        path = os.path.join(folder, file)
        print(f"📖 Extracting from {file}...")
        
        try:
            with pdfplumber.open(path) as pdf:
                file_text = ""
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Clean up text: replace multiple spaces and newlines
                        cleaned_text = " ".join(text.split())
                        file_text += cleaned_text + " "
                    
                    if page_num % 10 == 0:
                        print(f"   Processing page {page_num}/{len(pdf.pages)}...")
                
                if file_text.strip():
                    all_text += f"\n\n=== SOURCE: {file} ===\n\n{file_text}\n"
                    successful_extractions += 1
                    print(f"   ✅ Extracted {len(file_text)} characters from {file}")
                else:
                    print(f"   ⚠️ Warning: No text found in {file} (might be a scanned image)")
        
        except Exception as e:
            print(f"   ❌ Error processing {file}: {str(e)}")
    
    return all_text, successful_extractions, len(pdf_files)

if __name__ == "__main__":
    print("🚀 Starting text extraction from O/L textbooks...\n")
    
    text, success_count, total_count = extract_text_from_pdfs("textbooks")
    
    if not text.strip():
        print("\n❌ No text could be extracted from any PDFs!")
        print("Your PDFs might be scanned images. You'll need OCR (pytesseract) for those.")
        sys.exit(1)
    
    # Save extracted text
    output_file = "ol_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Successfully processed: {success_count}/{total_count} PDFs")
    print(f"   Total characters extracted: {len(text):,}")
    print(f"   Saved to: {output_file}")
    print(f"\n➡️  Next step: Run 'python build_index.py'")