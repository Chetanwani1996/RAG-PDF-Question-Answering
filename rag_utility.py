# """RAG utility functions for document processing and retrieval."""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

embedding = HuggingFaceEmbeddings()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

def process_document_to_chroma_db(file_name):
    # load the PDF document using unstructured loader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    # split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)

    # store the document chunks in a Chroma vector database

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    return 0

def answer_question(user_question):
    # load the Chroma vector database
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    retriever = vectordb.as_retriever()

    # create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    # get the answer to the query
    response = qa_chain.invoke({"query": user_question})
    result = response["result"]

    return result