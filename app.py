import os

import streamlit as st
from dotenv import load_dotenv
from rag_utility import process_document_to_chroma_db, answer_question

# Load environment variables
load_dotenv()

#set the working directory
working_dir = os.getcwd()

st.title("RAG Document Question Answering")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # define the file path to save the uploaded PDF
    saved_file_path = os.path.join(working_dir, uploaded_file.name)

    # save the uploaded PDF to the defined file path

    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("🔄 Processing document..."):
            process_document = process_document_to_chroma_db(uploaded_file.name)

        st.success("✅ Document processed and stored in Chroma vector database. You can now ask questions about the document.")

        user_question = st.text_input("Ask a question about the document:")

        if st.button("Get Answer"):
            if user_question:
                with st.spinner("🤔 Generating answer..."):
                    answer = answer_question(user_question)
                
                st.markdown("**Answer:**")
                st.markdown(answer)
            else:
                st.warning("Please enter a question before clicking the button.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Troubleshooting: Check your .env file has GROQ_API_KEY and all dependencies are installed")