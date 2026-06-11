import os
import tempfile
import warnings

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("PDF Chatbot")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.2,
        reasoning_format="parsed",
    )


@st.cache_resource
def build_rag_chain(pdf_bytes):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as tmp:

        tmp.write(pdf_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful PDF assistant.
                
                Answer ONLY from the provided context.
                
                If the answer is not present in the PDF,
                reply with:
                
                "I could not find that information in the PDF."
                
                <context>
                {context}
                </context>
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = get_llm()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    return retrieval_chain


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], max_upload_size=15)

if uploaded_file is None:
    st.info("Upload a PDF to start chatting.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

with st.spinner("Processing PDF..."):
    retrieval_chain = build_rag_chain(uploaded_file.getvalue())

for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


query = st.chat_input("Ask something about the PDF...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        response = retrieval_chain.invoke(
            {
                "input": query,
                "chat_history": st.session_state.chat_history,
            }
        )

        answer = response["answer"]

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))

    with st.expander("Retrieved Chunks"):

        for idx, doc in enumerate(
            response["context"],
            start=1,
        ):
            st.markdown(f"### Chunk {idx}")
            page = doc.metadata.get(
                "page",
                "Unknown",
            )
            st.write(f"**Page:** {page + 1 if isinstance(page, int) else page}")
            st.write(doc.page_content[:1000])
