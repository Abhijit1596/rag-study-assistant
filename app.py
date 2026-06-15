from src.chatbot import ask_question
from src.embeddings import create_vector_store
from src.pdf_processor import extract_text_from_pdf
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Study Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_text = ""

    for uploaded_file in uploaded_files:

        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        text = extract_text_from_pdf(
            uploaded_file.name
        )

        all_text += text + "\n"

    index, chunks, model = create_vector_store(
        all_text
    )

    col1, col2 = st.columns(2)

    # PDF SUMMARY
    with col1:
        if st.button("📄 Generate PDF Summary"):

            summary_question = "summarize the entire document in detail"

            summary, source = ask_question(
                summary_question,
                index,
                chunks,
                model
            )

            st.subheader("📄 PDF Summary")
            st.markdown(summary)

    # IMPORTANT QUESTIONS
    with col2:
        if st.button("📝 Important Questions"):

            questions_prompt = "generate important exam questions from this document"

            questions, source = ask_question(
                questions_prompt,
                index,
                chunks,
                model
            )

            st.subheader("📝 Important Questions")
            st.markdown(questions)

    # CHAT
    question = st.chat_input(
        "Ask a question from your PDFs"
    )

    if question:

        answer, source = ask_question(
            question,
            index,
            chunks,
            model
        )

        st.session_state.chat_history.append(
            ("user", question)
        )

        st.session_state.chat_history.append(
            ("assistant", answer)
        )

# CHAT HISTORY
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)
