import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

DATA_DIR = Path(__file__).parent / "data"
MODEL = "gemini-3.8-flash"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

load_dotenv()


def get_api_key():
    """Return the Gemini API key from .env, or stop the app with a fix-it message."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your-key":
        st.error(
            "GEMINI_API_KEY is missing. Add a line like `GEMINI_API_KEY=AIza...` "
            "to the .env file in the project folder, then restart the app."
        )
        st.stop()
    return api_key


def check_data_dir():
    """Stop the app if DATA_DIR is missing, not a folder, or has no visible files."""
    if not DATA_DIR.is_dir():
        st.error(
            f"Data folder not found: `{DATA_DIR}`. Create a folder named `data` "
            "next to app.py and put the handbook PDF in it."
        )
        st.stop()

    files = [f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        st.error(f"The data folder `{DATA_DIR}` is empty. Add the handbook PDF, then restart the app.")
        st.stop()


@st.cache_resource
def get_index(api_key):
    """Load the documents and build the vector index once, then reuse it on every rerun."""
    Settings.llm = GoogleGenAI(model=MODEL, api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    return VectorStoreIndex.from_documents(documents)


def main():
    """Validate setup, build the engine, and run the chat UI."""
    st.title("Babson Handbook Chatbot")

    api_key = get_api_key()
    check_data_dir()

    try:
        with st.spinner("Loading the handbook..."):
            index = get_index(api_key)
    except genai_errors.ClientError as e:
        st.error(f"Gemini rejected the request. Check that your API key and model name are valid. Details: {e}")
        st.stop()
    except httpx.TransportError:
        st.error("Couldn't reach Gemini. Check your internet connection and restart the app.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't build the chatbot (corrupt file or failed model download?). Details: {e}")
        st.stop()

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
            try:
                with st.spinner("Searching..."):
                    answer = str(st.session_state.chat_engine.chat(prompt))
            except httpx.TransportError:
                st.error("Connection lost. Check your internet and ask again.")
                return
            except genai_errors.ClientError as e:
                if e.code == 429:
                    st.error("Rate limit reached. Wait a minute and try again.")
                else:
                    st.error(f"Gemini couldn't answer that request. Details: {e}")
                return
            except genai_errors.ServerError:
                st.error("Gemini is having trouble right now. Try again in a moment.")
                return
            except Exception as e:
                st.error(f"Something went wrong answering that question. Try again. Details: {e}")
                return
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


main()