import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama
import os

st.set_page_config(
    page_title="O/L Study Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load the vector store and model (cached for performance)."""
    if not os.path.exists("vectorstore.pkl"):
        st.error("❌ Vector store not found! Please run build_index.py first.")
        st.stop()
    
    with open("vectorstore.pkl", "rb") as f:
        data = pickle.load(f)
    
    model = SentenceTransformer(data['model_name'])
    
    return data['chunks'], data['index'], model

def retrieve_context(query, chunks, index, model, top_k=3):
    """Retrieve relevant context for a query."""
    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype('float32')
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

def get_answer(query, chunks, index, model):
    """Get an answer from the chatbot."""
    try:
        context_chunks = retrieve_context(query, chunks, index, model, top_k=3)
        context = "\n\n".join(context_chunks)
        
        prompt = f"""You are a helpful tutor for O/L students in Sri Lanka.

Textbook Content:
{context}

Question: {query}

Provide a clear, concise answer suitable for O/L students. If the context doesn't contain enough information, say so honestly.

Answer:"""
        
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.7}
        )
        
        return response['message']['content']
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main app
def main():
    st.markdown('<h1 class="main-header">G.C.E. O/L Study Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Made by Sandeep Vaz</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your textbooks - 100% offline & free</p>', unsafe_allow_html=True)
    
    # Load chatbot
    with st.spinner("Loading chatbot..."):
        chunks, index, model = load_chatbot()
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask a question about your textbooks..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(query, chunks, index, model)
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(f"""
        This chatbot uses:
        - **{len(chunks):,}** text chunks from your textbooks
        - **Semantic search** to find relevant content
        - **Llama3** AI model for generating answers
        
        All processing happens **locally** on your computer.
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("Made with ❤️ for O/L students - Sandeep Vaz")

if __name__ == "__main__":
    main()