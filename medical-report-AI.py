import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ✅ Set Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC5hFpd3IWzM3MT32RqFjUx1RReeHrHIM4"

# ✅ Streamlit Page Config
st.set_page_config(page_title="🩺 Medical Report Chatbot", layout="wide")
st.title("🩺 AI Medical Report Analyzer")

# ✅ Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# 📥 File Reader
def read_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text.strip()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            return "❌ Unsupported file type."
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"

# ⚙️ Process the Uploaded File
def process_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext not in ['pdf', 'txt']:
        return "❌ Only PDF or TXT files are supported."

    file_path = f"temp_uploaded.{ext}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = read_file(file_path)
    if not text or text.startswith("❌"):
        return text

    # Vectorizing content
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    # Gemini LLM setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )

    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return "✅ File processed! Ask questions below."

# 💬 Display Chat Messages
def show_chat():
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

# 📤 File Uploader
uploaded_file = st.file_uploader("📄 Upload your medical report (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    status = process_file(uploaded_file)
    if "✅" in status:
        st.success(status)
    else:
        st.error(status)

# 🧠 Chatbot Interface
if st.session_state.qa_chain:
    show_chat()
    user_input = st.chat_input("Ask a question about your medical report...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        response = st.session_state.qa_chain.run(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append((user_input, response))
else:
    st.info("Please upload a medical report to get started.")
