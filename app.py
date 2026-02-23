import os
import streamlit as st
from dotenv import load_dotenv
from rag_utility import process_document_to_chroma_db, answer_question

# Load environment variables
load_dotenv()

# Set working directory
working_dir = os.getcwd()

st.set_page_config(page_title="RAG PDF QA", page_icon="📄")
st.title("📄 RAG Document Question Answering")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    saved_file_path = os.path.join(working_dir, uploaded_file.name)

    # Save uploaded file
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("🔄 Processing document..."):
            process_document_to_chroma_db(uploaded_file.name)

        st.success("✅ Document processed and stored in Chroma DB.")

        user_question = st.text_input("Ask a question about the document:")

        if st.button("Get Answer"):
            if user_question.strip() != "":
                with st.spinner("🤔 Generating answer..."):
                    answer = answer_question(user_question)

                st.markdown("### 📌 Answer")
                st.write(answer)
            else:
                st.warning("Please enter a question.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure GROQ_API_KEY is set in your .env file.")
