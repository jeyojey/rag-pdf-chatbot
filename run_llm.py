import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Color palette
background_color = "#5A189A"  # Dark violet
text_color = "#FFD700"  # Gold (Yellow)
user_message_color = "#7B2CBF"  # Light Violet
ai_message_color = "#FFC300"  # Golden Yellow

# Custom CSS for styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {text_color};
        color: {background_color};
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {text_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        background-color: {background_color};
        color: {text_color};
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {text_color};
        color: {background_color};
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .chat-message {{
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 8px;
        font-size: 16px;
    }}
    .user {{
        background-color: {user_message_color};
        color: white;
        text-align: right;
    }}
    .ai {{
        background-color: {ai_message_color};
        color: {background_color};
        text-align: left;
    }}
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("Chat with Your PDF (RAG System)")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the PDF
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", ".", "?", "!"]
    )
    documents = text_splitter.split_documents(docs)

    # Embed the documents
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = FAISS.from_documents(documents, embedder)

    # Define retriever with similarity search
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define the LLM
    llm = Ollama(model="deepseek-r1")

    # Define the prompt
    prompt = """
    You are an AI assistant answering questions based on the provided context.
    Use ONLY the context below to answer. If the answer is not found, say "I don't know."

    Context:
    {context}

    Question:
    {question}

    Answer:
    - Provide a clear and informative response.
    - If possible, cite the document's source.
    - Keep responses detailed but concise.
    """

    # Initialize the RAG system
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # User input for chat
    user_input = st.text_input("Ask a question about the PDF:")

    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "text": user_input})

        # Get response from the model
        with st.spinner("Thinking..."):
            response = qa(user_input)["result"]

        # Store AI response
        st.session_state.messages.append({"role": "ai", "text": response})

    # Display chat history in reverse order (newest first)
    for msg in reversed(st.session_state.messages):
        role, text = msg["role"], msg["text"]
        st.markdown(f'<div class="chat-message {role}">{text}</div>', unsafe_allow_html=True)

else:
    st.write("Please upload a PDF file to start chatting.")
