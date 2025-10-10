"""
Building a PDF chat bot — Retrieval Augmented Generation (RAG)

Reference: https://medium.com/@prithiviraj7r/building-a-pdf-chat-bot-retrieval-augmented-generation-rag-0bcf6060bbd6

Date: Monday - 6th October 2025

List of components:
• Extract text from PDF docs
• Chunking/Segmentation of text
• Embedding text & Ingestion in Vectorstore
• Conversation using LLMs (OpenAI)
"""
#Prompt template to add persona to the bot
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_template(
    """You are a senior digital marketer.
       You are concise, practical, and give campaign-level suggestions: target audience, messaging, ad formats, copy variants, testing plan, and expected KPIs. 
       Use the retrieved past campaign data as evidence and suggest the best approach for the new campaign.
    """)

from PyPDF2 import PdfReader, PdfWriter

def get_pdf_content(pdf_paths):

    raw_text = ""
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        content = []
        for page in reader.pages:
            content.append(page.extract_text())
    
    return "\n".join(content)

# Chunking

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        text_length = len(text)
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Embeddings and Vector Store

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Fast and Free
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store
# Conversation using LLMs (OpenAI)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(
        repo_id = "HuggingFaceH4/zephyrl-mini",
        temperature=0, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

import streamlit as st

from web_template import css, bot_template, user_template

def process_query(query_text):
    response = st.session_state.conversation_chain({'question': query_text})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            #st.markdown(f"**User:** {message.content}")
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            #st.markdown(f"**Bot:** {message.content}")
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    return response
from dotenv import load_dotenv

def main():
    load_dotenv()

    st.set_page_config(page_title="RAG Chatbot", page_icon=":robot_face:", layout="wide")
    #st.image("templates/logo.png", width=100)

    st.write(css, unsafe_allow_html=True)

    st.header("RAG Chatbot :robot_face:")
    query = st.text_input("Ask a question about your document:", placeholder="Type your question here...", key="input")

    if query:
        process_query(query)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    with st.sidebar:
        st.subheader("Your Document")
        pdf = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

        if st.button("Run"):
            with st.spinner("Processing..."):
                # extract text from pdf documents
                raw_text = get_pdf_content(pdf)

                # convert text to chunks of data
                chunks = chunk_text(raw_text)

                # create vector embeddings
                vector_store = get_embeddings(chunks)

                # create conversation chain
                st.session_state.conversation_chain = get_conversation_chain(vector_store)

            st.success("Document processed successfully!")

if __name__ == "__main__":
    main()
