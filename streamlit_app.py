import streamlit as st
from openai import OpenAI
import fitz
import faiss
from langchain_huggingface import HuggingFaceEmbeddings          
from langchain_community.vectorstores import FAISS               
from langchain.schema import Document
from io import StringIO
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"  # lekki, dobry jakościowo
model_kwargs   = {"device": "cpu", "trust_remote_code": True}
embeddings     = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs=model_kwargs
)


def create_index(documents):
    embeddings = ... #wybor modelu
    texts = ... #wartosci tekstowe wszystkich dokumentów
    metadata = ... #metadata wszystkich dokumentów czyli slownik {filename:...}

    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    #index = faiss. ... #ustawienie indexu przeszukania
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def load_pdf(file_path):
    doc=fitz.open(file_path)
    text=""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(folder_path, filename))
            documents.append({"filename": filename, "text": text})
        return documents

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_content = ""
        for page in doc:
            file_content += page.get_text()
        doc.close()
        st.success(f"Wczytano: {uploaded_file.name}")
        st.success(f"Tekst: {file_content}")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model= selected_model,
        messages=st.session_state.messages 
        )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

