# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os


# app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# embeddings = download_hugging_face_embeddings()

# index_name = "medical-chatbot" 

# # Embed each chunk and upsert the embeddings into Pinecone index
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# chatModel = ChatOpenAI(
#     model="meta-llama/llama-3-8b-instruct:free",
#     openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1",
#     temperature=0
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8080, debug= True)

import streamlit as st
import os
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

# 1. Setup Streamlit Page Settings
st.set_page_config(page_title="Medical Chatbot", page_icon="⚕️", layout="centered")
st.title("⚕️ Medical Chatbot")

# 2. Handle Secrets securely for Streamlit Cloud
# This attempts to get keys from Streamlit secrets first, then falls back to local environment variables.
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.environ.get("PINECONE_API_KEY"))
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY"))

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# 3. Cache the LangChain / RAG Initialization
# This ensures we don't re-download embeddings or re-initialize connections on every chat message.
@st.cache_resource
def initialize_rag_chain():
    embeddings = download_hugging_face_embeddings()
    index_name = "medical-chatbot" 

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    chatModel = ChatOpenAI(
        model="meta-llama/llama-3-8b-instruct:free",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

rag_chain = initialize_rag_chain()

# 4. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input and Logic
if user_input := st.chat_input("Ask a medical question..."):
    # Display user's message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            st.markdown(answer)
    
    # Add assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": answer})