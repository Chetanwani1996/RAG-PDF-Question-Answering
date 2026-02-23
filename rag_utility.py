import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA

# Load environment variables
load_dotenv()

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


def process_document_to_chroma_db(file_name):
    """
    Load PDF, split into chunks, and store in Chroma vector DB.
    """

    file_path = os.path.join(working_dir, file_name)

    # Load PDF
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    split_docs = text_splitter.split_documents(documents)

    # Store in Chroma DB
    Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore")
    )


def answer_question(user_question):
    """
    Retrieve relevant chunks and generate answer using Groq.
    """

    # Load Chroma DB
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore")
    )

    retriever = vectordb.as_retriever()

    # Create Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    response = qa_chain.invoke({"query": user_question})

    return response["result"]
