import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def process_document_to_chroma_db(file_name):
    file_path = os.path.join(working_dir, file_name)

    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    split_docs = text_splitter.split_documents(documents)

    Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore")
    )


def answer_question(user_question):
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore")
    )

    retriever = vectordb.as_retriever()

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_question})

    return response["answer"]
