O/L Study Chatbot - AI-Powered Textbook Assistant

An intelligent, offline chatbot that helps Sri Lankan O/L students by answering questions from their textbooks using Retrieval-Augmented Generation (RAG) and local LLMs.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🎯 Features

- **📚 Multi-Subject Support**: Works with any O/L textbook in PDF format
- **🌐 Multilingual**: Supports both English and Sinhala (සිංහල) content
- **🔒 100% Offline & Private**: All processing happens locally, no data sent to cloud
- **💰 Completely Free**: Uses open-source tools and local AI models
- **⚡ Fast Semantic Search**: FAISS-powered vector search for relevant content
- **🤖 Smart AI Responses**: Llama3-powered natural language understanding
- **🖥️ Dual Interface**: Command-line and web-based (Streamlit) interfaces

## 🏗️ Architecture

```
┌─────────────────┐
│  PDF Textbooks  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Extraction│  (pdfplumber)
│  & Chunking     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector         │  (sentence-transformers)
│  Embeddings     │  paraphrase-multilingual-MiniLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │  (faiss-cpu)
│  Index          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Question       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Semantic       │  (cosine similarity)
│  Search         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context +      │
│  Prompt         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Llama3 LLM     │  (Ollama)
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Answer         │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (for running Llama3)
- [Ollama](https://ollama.com/download) installed

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/msnvaz/ol_chatbot.git
cd ol-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama and download model**
```bash
ollama pull llama3
```

4. **Add your textbooks**
```bash
mkdir textbooks
# Copy your O/L textbook PDFs to the textbooks/ folder
```

### Usage

**Step 1: Extract text from PDFs**
```bash
python extract_text.py
```

**Step 2: Build vector index**
```bash
python build_index.py
```

**Step 3: Start chatting**

*Command-line interface:*
```bash
python chatbot.py
```

*Web interface:*
```bash
streamlit run streamlit_app.py
```

## 📊 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Parsing** | pdfplumber, PyPDF2 | Extract text from textbooks |
| **Text Chunking** | Custom algorithm | Split text with overlap for context |
| **Embeddings** | sentence-transformers | Convert text to semantic vectors |
| **Vector DB** | FAISS | Fast similarity search |
| **LLM** | Llama3 (via Ollama) | Generate natural language answers |
| **Web UI** | Streamlit | Interactive chat interface |

## 🎓 How It Works

1. **Text Extraction**: PDFs are parsed and text is extracted with proper Unicode handling for Sinhala
2. **Chunking**: Text is split into 300-word chunks with 50-word overlap to maintain context
3. **Embedding**: Each chunk is converted to a 384-dimensional vector using multilingual model
4. **Indexing**: Vectors are stored in FAISS for efficient similarity search
5. **Query Processing**: User questions are embedded and matched against chunk vectors
6. **Context Retrieval**: Top-k most relevant chunks are retrieved (default: 3)
7. **Answer Generation**: Retrieved context + question sent to Llama3 for natural response

## 📁 Project Structure

```
ol_chatbot/
├── textbooks/              # PDF textbooks (gitignored)
├── extract_text.py         # PDF text extraction
├── build_index.py          # Vector index creation
├── chatbot.py              # CLI chatbot
├── streamlit_app.py        # Web interface
├── requirements.txt        # Python dependencies
├── .gitignore             
└── README.md
```

## 🔧 Configuration

### Adjust Chunk Size
In `build_index.py`:
```python
chunks = chunk_text_smart(text, chunk_size=300, overlap=50)
# Smaller chunks = more precise, less context
# Larger chunks = more context, less precise
```

### Change Retrieval Count
In `chatbot.py`:
```python
context_chunks = self.retrieve_context(query, top_k=3)
# Increase for more context (slower)
# Decrease for faster responses
```

### Switch AI Model
```bash
# Use faster model
ollama pull phi3:mini

# In chatbot.py, change:
model='phi3:mini'
```

## 🌟 Key Features & Innovations

- **Multilingual Support**: Custom implementation supporting both English and Sinhala
- **Smart Chunking**: Overlapping chunks preserve context across boundaries
- **Fallback Mechanisms**: Multiple PDF parsers for reliability
- **Error Handling**: Comprehensive try-catch blocks with helpful error messages
- **Memory Efficient**: Lazy loading and caching strategies
- **User-Friendly**: Progress indicators and clear status messages

## 📈 Performance

- **Text Extraction**: ~1-2 seconds per page
- **Index Building**: ~5-10 minutes for 500 pages
- **Query Response**: 10-30 seconds (first query: 30-60s for model loading)
- **Vector Search**: <100ms for similarity matching

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for multilingual embeddings
- [Ollama](https://ollama.com/) for local LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Streamlit](https://streamlit.io/) for rapid UI development

## 📧 Contact

LinkedIn - [Sandeep Vaz](https://www.linkedin.com/in/sandeep-vaz-447662283/)

Project Link: [https://github.com/msnvaz/ol_chatbot.git](https://github.com/msnvaz/ol_chatbot.git)

---