from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import sys
import numpy as np

def chunk_text_smart(text, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks.
    Overlap helps maintain context between chunks.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only keep meaningful chunks
            chunks.append(chunk.strip())
    
    return chunks

def build_vector_index():
    """Create embeddings and FAISS index from extracted text."""
    
    # Check if text file exists
    if not os.path.exists("ol_text.txt"):
        print("❌ Error: ol_text.txt not found!")
        print("Please run 'python extract_text.py' first.")
        sys.exit(1)
    
    print("📖 Loading extracted text...")
    with open("ol_text.txt", encoding="utf-8") as f:
        text = f.read()
    
    if len(text.strip()) < 100:
        print("❌ Error: Text file is too small or empty!")
        sys.exit(1)
    
    print(f"   Total text length: {len(text):,} characters\n")
    
    # Split into chunks
    print("✂️  Splitting text into chunks...")
    chunks = chunk_text_smart(text, chunk_size=300, overlap=50)
    print(f"   Created {len(chunks):,} chunks\n")
    
    if len(chunks) == 0:
        print("❌ Error: No chunks created!")
        sys.exit(1)
    
    # Load embedding model
    print("🤖 Loading sentence transformer model...")
    print("   (This may take a minute on first run - downloading model)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ Model loaded\n")
    
    # Create embeddings
    print("🔢 Creating embeddings for all chunks...")
    print("   (This may take a few minutes depending on text size)")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype('float32')
    print(f"   ✅ Created embeddings: {embeddings.shape}\n")
    
    # Build FAISS index
    print("🗄️  Building FAISS vector index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"   ✅ Index built with {index.ntotal} vectors\n")
    
    # Save everything
    print("💾 Saving vector store...")
    data = {
        'chunks': chunks,
        'index': index,
        'model_name': 'all-MiniLM-L6-v2'
    }
    
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize("vectorstore.pkl") / (1024 * 1024)
    print(f"   ✅ Vector store saved to vectorstore.pkl ({file_size:.2f} MB)\n")
    
    print("🎉 Vector index created successfully!")
    print(f"\n➡️  Next step: Run 'python chatbot.py' to start chatting")

if __name__ == "__main__":
    print("🚀 Building vector index for O/L chatbot...\n")
    build_vector_index()