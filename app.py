
import streamlit as st
import chromadb
import ollama

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from html import escape


# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="RAG Agent",
    page_icon="📚",
    layout="centered"
)


# -------------------------------
# DARK THEME DESIGN
# -------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b0b12;
    color: white;
}

.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
}

h1 {
    color: #a78bfa !important;
    text-align: center;
}

h2, h3 {
    color: #c4b5fd !important;
}

p, label {
    color: #e2e8f0 !important;
}

[data-testid="stFileUploader"] {
    background-color: #171724;
    border: 1px solid #6d5dfc;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stTextInput"] input {
    background-color: #1e1e2e !important;
    color: white !important;
    border: 1px solid #6d5dfc !important;
    border-radius: 10px !important;
}

.stButton button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    width: 100%;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
}

.stButton button:hover {
    border: 1px solid white;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -------------------------------
# CHAT HISTORY
# -------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------
# MAIN TITLE
# -------------------------------

st.title("📚 RAG Agent")
st.markdown(
    "<p style='text-align:center;'>"
    "Ask questions from your PDF documents "
    "and get AI answers with citations."
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# -------------------------------
# PDF UPLOAD
# -------------------------------

st.subheader("📂 Upload Your PDF Documents")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# -------------------------------
# QUESTION INPUT
# -------------------------------

st.subheader("💬 Ask Your Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is Elliptic Curve Encryption?"
)


# -------------------------------
# SETTINGS
# -------------------------------

MAX_DISTANCE = 1.0
CHUNK_SIZE = 500
TOP_K = 3
OLLAMA_MODEL = "qwen2.5:1.5b"


# -------------------------------
# PROCESS BUTTON
# -------------------------------

if st.button("Ask AI"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:

        # -------------------------------
        # EXTRACT TEXT FROM ALL PDF FILES
        # -------------------------------

        all_chunks = []
        all_metadatas = []

        with st.spinner("Reading your PDF documents..."):

            for uploaded_file in uploaded_files:

                uploaded_file.seek(0)
                pdf = PdfReader(uploaded_file)

                for page_number, page in enumerate(
                    pdf.pages, start=1
                ):

                    page_text = page.extract_text()

                    if not page_text:
                        continue

                    page_text = page_text.strip()

                    for i in range(
                        0, len(page_text), CHUNK_SIZE
                    ):

                        chunk = page_text[
                            i:i + CHUNK_SIZE
                        ].strip()

                        if chunk:

                            all_chunks.append(chunk)

                            all_metadatas.append({
                                "source": uploaded_file.name,
                                "page": page_number
                            })

        # -------------------------------
        # CHECK EXTRACTED TEXT
        # -------------------------------

        if not all_chunks:

            st.error(
                "No readable text was found in the uploaded PDFs."
            )

        else:

            # -------------------------------
            # CREATE EMBEDDINGS
            # -------------------------------

            with st.spinner("Understanding your documents..."):

                document_embeddings = embedding_model.encode(
                    all_chunks,
                    normalize_embeddings=True
                )

            # -------------------------------
            # STORE IN CHROMADB
            # -------------------------------

            client = chromadb.EphemeralClient()

            collection = client.create_collection(
                name="college_notes",
                metadata={"hnsw:space": "cosine"}
            )

            ids = [
                f"chunk_{i}"
                for i in range(len(all_chunks))
            ]

            collection.add(
                ids=ids,
                documents=all_chunks,
                embeddings=document_embeddings.tolist(),
                metadatas=all_metadatas
            )

            # -------------------------------
            # SEARCH RELEVANT INFORMATION
            # -------------------------------

            with st.spinner("Searching your documents..."):

                query_embedding = embedding_model.encode(
                    question,
                    normalize_embeddings=True
                )

                results = collection.query(
                    query_embeddings=[
                        query_embedding.tolist()
                    ],
                    n_results=min(TOP_K, len(all_chunks))
                )

            retrieved_chunks = results["documents"][0]
            retrieved_metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            relevant_chunks = []
            relevant_sources = []

            for chunk, metadata, distance in zip(
                retrieved_chunks,
                retrieved_metadatas,
                distances
            ):

                if distance <= MAX_DISTANCE:

                    relevant_chunks.append(chunk)
                    relevant_sources.append(metadata)

            # -------------------------------
            # GENERATE AI ANSWER
            # -------------------------------

            if not relevant_chunks:

                answer = (
                    "I couldn't find this information in your PDF."
                )

            else:

                context = ""

                for chunk, source in zip(
                    relevant_chunks,
                    relevant_sources
                ):

                    context += (
                        f"\nSource: {source['source']}\n"
                        f"Page: {source['page']}\n"
                        f"Content: {chunk}\n"
                    )

                prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the information
provided in the PDF context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context,
   reply exactly:
   I couldn't find this information in your PDF.
4. Give a simple and clear answer.
5. Do not create fake citations.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

                with st.spinner("Generating AI answer..."):

                    try:

                        response = ollama.chat(
                            model=OLLAMA_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        )

                        answer = response["message"]["content"]

                    except Exception as e:

                        answer = None
                        st.error(
                            "Could not connect to Ollama. "
                            "Please check that Ollama is running "
                            "and the model is installed."
                        )
                        st.caption(str(e))

            # -------------------------------
            # DISPLAY ANSWER
            # -------------------------------

            if answer is not None:

                st.divider()
                st.subheader("🤖 AI Answer")

                with st.container(border=True):

                    st.write(answer)

                # -------------------------------
                # DISPLAY CITATIONS
                # -------------------------------

                if relevant_sources:

                    st.subheader("📖 Sources")

                    unique_sources = []

                    for source in relevant_sources:

                        citation = (
                            f"{source['source']} - "
                            f"Page {source['page']}"
                        )

                        if citation not in unique_sources:
                            unique_sources.append(citation)

                    for citation in unique_sources:

                        with st.container(border=True):
                            st.write("📄 " + citation)

                else:

                    st.info(
                        "No relevant PDF source was found."
                    )

                # -------------------------------
                # SAVE CHAT HISTORY
                # -------------------------------

                citation_text = ", ".join(
                    unique_sources
                ) if relevant_sources else "No source found"

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "citation": citation_text
                })


# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------

st.divider()
st.subheader("💬 Chat History")

if st.session_state.chat_history:

    for chat in reversed(st.session_state.chat_history):

        with st.container(border=True):

            st.markdown("**You:**")
            st.write(chat["question"])

            st.markdown("**AI Assistant:**")
            st.write(chat["answer"])

            st.caption("Sources: " + chat["citation"])

else:

    st.caption("Your questions and answers will appear here.")
