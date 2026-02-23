# FileBot - RAG Document Q&A System

A Streamlit application for asking questions about PDF documents using Retrieval Augmented Generation (RAG) powered by LangChain and Groq API.

## Features

- 📄 **PDF Upload & Processing** - Upload and process PDF documents
- 🤖 **AI-Powered Q&A** - Ask questions about your documents using LLMs
- 🧮 **Vector Embeddings** - HuggingFace embeddings for semantic search
- 🗂️ **Persistent Storage** - Chroma vector database for document storage
- ⚡ **Fast Inference** - Powered by Groq's fast LLM inference
- 🎨 **User-Friendly Interface** - Clean Streamlit web interface

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- pip (Python package manager)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

1. Copy `env_template.txt` to `.env`:
   ```bash
   cp env_template.txt .env
   ```

2. Add your API keys to `.env`:
   - Get GROQ_API_KEY from [Groq Console](https://console.groq.com)
   - (Optional) Get HuggingFace token if using private models

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Usage

1. **Upload Document**: Click on the file uploader in the sidebar to select a PDF
2. **Process**: Click "Process Document" to extract and index the content
3. **Ask Questions**: Type your question in the query area
4. **Get Answers**: Click "Search" to get AI-powered answers based on your document

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── rag_utility.py         # RAG utility functions
├── requirements.txt       # Python dependencies
├── env_template.txt       # Environment variables template
└── README.md             # This file
```

## How RAG Works

1. **Document Loading** - PDF files are loaded using UnstructuredPDFLoader
2. **Text Splitting** - Documents are split into manageable chunks
3. **Embedding** - Text chunks are converted to embeddings using HuggingFace models
4. **Vector Storage** - Embeddings are stored in Chroma vector database
5. **Retrieval** - User queries retrieve relevant document chunks
6. **Generation** - LLM generates answers based on retrieved content

## Technologies Used

- **Streamlit** - Web application framework
- **LangChain** - LLM orchestration and chains
- **Groq API** - Fast LLM inference
- **HuggingFace** - Text embeddings
- **Chroma** - Vector database
- **Unstructured** - PDF document loading

## Troubleshooting

### Issue: "GROQ_API_KEY not found"
- Ensure `.env` file exists in the project root
- Verify GROQ_API_KEY is set correctly in `.env`

### Issue: "PDF processing fails"
- Ensure the PDF file is not corrupted
- Try with a different PDF file
- Check available disk space

### Issue: "Slow response times"
- This is expected for the first query (model loading)
- Subsequent queries should be faster
- Consider using a smaller document or fewer chunks

## API Keys

### Getting Groq API Key
1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in
3. Create a new API key
4. Add it to your `.env` file

## Notes

- The application uses the Llama 3.3 70B model by default (via Groq)
- Vector stores are persisted in the `./chroma_db` directory
- Temporary PDF files are cleaned up after processing
- Session state is maintained during the Streamlit session

## License

This project is open source and available for personal and commercial use.

## Support

For issues, questions, or contributions, please create an issue or submit a pull request.



