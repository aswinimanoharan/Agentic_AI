import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
#from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.document_loaders import PyPDFLoader

import tempfile

# ------------------------ SET API KEY ------------------------
GEMINI_API_KEY = "AIzaSyC4KWzVoB8uGmiOQBqldPC-PoChlIN_8KY"  # Replace with your actual key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="📄 RAG with LangChain + Gemini", layout="wide")
st.title("📄 RAG App with Conversational Memory (LangChain + Gemini)")

# ------------------------ UPLOAD FILE ------------------------
uploaded_file = st.file_uploader("📤 Upload a PDF or Text file", type=["pdf", "txt"])

if uploaded_file:
    st.success(f"✅ {uploaded_file.name} uploaded successfully")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # ------------------------ LOAD DOCUMENT ------------------------
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # ------------------------ SPLIT DOCUMENT ------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # ------------------------ EMBEDDING MODEL ------------------------
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ------------------------ VECTORSTORE ------------------------
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ------------------------ MEMORY ------------------------
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # ------------------------ RETRIEVER ------------------------
    retriever = vectorstore.as_retriever()

    # ------------------------ LLM ------------------------
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    # ------------------------ CONVERSATIONAL CHAIN ------------------------
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    # ------------------------ CHAT INTERFACE ------------------------
    st.subheader("💬 Ask Questions Based on Your File")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("❓ Your Question")

    if st.button("💡 Ask"):
        if user_question:
            try:
                result = qa_chain({"question": user_question})
                answer = result["answer"]

                # Append to chat history
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Gemini", answer))

                st.success("✅ Answer generated!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ------------------------ DISPLAY CHAT ------------------------
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🗣️ Chat History")
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {msg}")
