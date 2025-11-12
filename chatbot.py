import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import sys

try:
    import ollama
except ImportError:
    print("❌ Error: ollama package not installed!")
    print("Install it with: pip install ollama")
    sys.exit(1)

class OLChatbot:
    def __init__(self):
        """Initialize the chatbot by loading the vector store."""
        print("🤖 Initializing O/L Chatbot...\n")
        
        if not os.path.exists("vectorstore.pkl"):
            print("❌ Error: vectorstore.pkl not found!")
            print("Please run 'python build_index.py' first.")
            sys.exit(1)
        
        print("📚 Loading vector store...")
        with open("vectorstore.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.index = data['index']
        self.model = SentenceTransformer(data['model_name'])
        
        print(f"   ✅ Loaded {len(self.chunks):,} text chunks")
        print(f"   ✅ Vector index ready with {self.index.ntotal} vectors\n")
        
        # Test Ollama connection
        print("🔗 Testing Ollama connection...")
        try:
            ollama.list()
            print("   ✅ Ollama is running\n")
        except Exception as e:
            print(f"   ❌ Error: Cannot connect to Ollama!")
            print(f"   Make sure Ollama is installed and running.")
            print(f"   Error details: {str(e)}")
            sys.exit(1)
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve the most relevant text chunks for a query."""
        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        
        distances, indices = self.index.search(query_vec, top_k)
        
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        return relevant_chunks, distances[0]
    
    def ask(self, query):
        """Ask a question and get an answer from the chatbot."""
        try:
            # Retrieve relevant context
            context_chunks, distances = self.retrieve_context(query, top_k=3)
            context = "\n\n".join(context_chunks)
            
            # Create prompt for LLM
            prompt = f"""You are a helpful tutor for O/L (Ordinary Level) students in Sri Lanka. 
Answer questions clearly and concisely based on the textbook content provided.

Textbook Context:
{context}

Student Question: {query}

Instructions:
- Answer in simple, clear English suitable for O/L students
- Base your answer on the textbook content provided
- If the context doesn't contain enough information, say so honestly
- Keep explanations concise but complete
- Use examples when helpful

Answer:"""
            
            # Get response from Ollama
            response = ollama.chat(
                model='llama3',
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def chat_loop(self):
        """Run an interactive chat session."""
        print("=" * 60)
        print("🎓 O/L Study Chatbot - Ready to help!")
        print("=" * 60)
        print("\nTips:")
        print("  • Ask questions about topics from your textbooks")
        print("  • Type 'exit' or 'quit' to end the session")
        print("  • Type 'clear' to see this message again")
        print("\n" + "=" * 60 + "\n")
        
        while True:
            try:
                query = input("📝 Your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thanks for studying! Good luck with your O/Ls!")
                    break
                
                if query.lower() == 'clear':
                    print("\n" + "=" * 60)
                    print("🎓 O/L Study Chatbot")
                    print("=" * 60 + "\n")
                    continue
                
                print("\n🤔 Thinking...\n")
                answer = self.ask(query)
                print(f"💡 Answer:\n{answer}\n")
                print("-" * 60 + "\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Exiting... Good luck with your studies!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")

def main():
    try:
        chatbot = OLChatbot()
        chatbot.chat_loop()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()