import streamlit as st
from openai import OpenAI
import fitz
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings          
from langchain_community.vectorstores import FAISS               
from langchain_core.documents import Document as Document
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

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results


def create_index(documents):
    embeddings = embeddings # załadowanie modelu embeddingowego
    texts = [doc["text"] for doc in documents] # wartości tekstowe wszystkich dokumentów
    metadata = [{"filename": doc["filename"], "text": doc["text"]} for doc in documents] # metadane wszystkich dokumentów, czyli słownik {filename:... , text:...}

    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    index = faiss.indexFlatL2(embeddings_matrix.shape[1])# ustawienie indeksu przeszukwania
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = embeddings # załadowanie modelu embeddingowego
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32") # embeddowanie zapytania (query)
    results = faiss_index.similarity_search(query_embedding, k) # zwrócenie wyników przeuszkiwania
    return results

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
        documents = [{"filename": uploaded_file.name, "text": file_content}]
        st.session_state["faiss_index"] = create_index(documents)
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

