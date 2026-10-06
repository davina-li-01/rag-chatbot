import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

DATA_DIR = Path(__file__).parent / "data"

load_dotenv()

st.title("Babson Handbook Chatbot")


@st.cache_resource
def load_index():
    """Load the handbook from DATA_DIR and build a searchable vector index."""
    Settings.llm = GoogleGenAI(model="gemini-3.8-flash", api_key=os.getenv("GEMINI_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    return VectorStoreIndex.from_documents(docs)


index = load_index()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about the Babson handbook"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = str(st.session_state.chat_engine.chat(prompt))
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})