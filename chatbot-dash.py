#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
DocuSearch – Dash version
Chat with your documents (CSV, PDF, DOCX, TXT, …)

Key features:
  • Unified loaders
  • Token‑aware chunking
  • Vector‑store caching
  • Conversational memory
  • System prompt for consistent behavior
  • Spinner while waiting for answers
"""

# ---------- Imports ----------
import os
import base64
import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv, find_dotenv

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from flask import session

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# ---------- 1. Environment ----------
load_dotenv(find_dotenv(), override=True)

# ---------- 2. Global Caches ----------
VECTOR_CACHE: Dict[str, FAISS] = {}        # key -> FAISS store
SESSION_STATE: Dict[str, Dict] = {}        # session_id -> {"memory": ..., "qa_chain": ..., "retriever": ...}

# ---------- 3. Helper Functions ----------
def get_session_id() -> str:
    """Create/retrieve a unique ID for the browser session."""
    if "uid" not in session:
        session["uid"] = str(uuid.uuid4())
    return session["uid"]

def file_to_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def load_and_split(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        loader = CSVLoader(file_path)
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(str(file_path))
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        loader = UnstructuredFileLoader(str(file_path))  # fallback

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, length_function=len
    )
    return splitter.split_documents(docs)

def build_vector_store(docs, embeddings, key: str):
    """Cache FAISS stores globally to avoid recomputation."""
    if key not in VECTOR_CACHE:
        VECTOR_CACHE[key] = FAISS.from_documents(docs, embeddings)
    return VECTOR_CACHE[key]

def index_files(uploaded_files):
    """Build a retriever from all uploaded files."""
    all_docs, file_hashes = [], []

    for content, filename in uploaded_files:
        # content is "data:;base64,...."
        _, content_string = content.split(",", 1)
        file_bytes = base64.b64decode(content_string)
        file_hash = file_to_hash(file_bytes)
        file_hashes.append(file_hash)

        tmp_path = Path(tempfile.gettempdir()) / f"{file_hash}{Path(filename).suffix}"
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        try:
            docs = load_and_split(tmp_path)
            for d in docs:
                d.metadata["source"] = filename
            all_docs.extend(docs)
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

    docs_hash = hashlib.sha256("".join(sorted(file_hashes)).encode()).hexdigest()
    embeddings = OpenAIEmbeddings()  # reads OPENAI_API_KEY from env
    vector_store = build_vector_store(all_docs, embeddings, docs_hash)
    return vector_store.as_retriever(search_kwargs={"k": 4})

# ---------- 4. Prompt ----------
SYSTEM_PROMPT = (
    "You are **DocuSearch**, a highly‑skilled AI assistant.\n"
    "• Answer questions strictly using the content of the uploaded documents.\n"
    "• If the answer is not present, reply: “I’m not sure based on the provided documents.”\n"
    "• Be concise and cite sources when possible.\n"
)

prompt_template = PromptTemplate(
    template=(
        SYSTEM_PROMPT +
        "\nHere are relevant excerpts from the documents:\n{context}\n\n"
        "Question: {question}"
    ),
    input_variables=["context", "question"]
)

# ---------- 5. Dash App ----------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server  # for gunicorn / deployment
server.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid.uuid4()))

def make_sidebar():
    return dbc.Card(
        [
            html.H5("Document Search System", className="card-title mt-3"),
            dbc.Input(
                id="api-key-input",
                type="password",
                placeholder="OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY") or "",
                className="mb-3",
            ),
            dbc.Label("Model"),
            dcc.Dropdown(
                id="model-select",
                options=[
                    {"label": "gpt-3.5-turbo", "value": "gpt-3.5-turbo"},
                    {"label": "gpt-4o-mini", "value": "gpt-4o-mini"},
                    {"label": "gpt-4o", "value": "gpt-4o"},
                ],
                value="gpt-3.5-turbo",
                clearable=False,
                className="mb-3",
            ),
            dbc.Label("Temperature"),
            dcc.Slider(
                id="temperature-slider",
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.2,
                marks={0: "0", 0.5: "0.5", 1: "1"},
                className="mb-4",
            ),
            dbc.Button("Clear Chat", id="clear-chat", color="warning", className="mb-3"),
            html.Hr(),
            dcc.Upload(
                id="file-upload",
                children=html.Div(
                    ["Drag & Drop or ", html.A("Select Files")],
                    style={"cursor": "pointer"},
                ),
                multiple=True,
                className="mb-3",
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
            ),
            html.Div(id="upload-status", className="text-success"),
        ],
        body=True,
        className="h-100",
    )

app.layout = dbc.Container(
    [
        dcc.Store(id="chat-history", data=[]),      # list of {"role": ..., "content": ...}
        dcc.Store(id="retriever-store"),            # will hold a dummy key; actual object in SESSION_STATE
        dcc.Store(id="qa-ready", data=False),       # bool flag
        dbc.Row(
            [
                dbc.Col(make_sidebar(), width=3),
                dbc.Col(
                    [
                        html.H2("📄 Chat With Your Documents"),
                        # Spinner wraps chat window
                        dcc.Loading(
                            id="loading-chat",
                            type="circle",
                            children=html.Div(
                                id="chat-window",
                                style={
                                    "height": "70vh",
                                    "overflowY": "auto",
                                    "border": "1px solid #ddd",
                                    "padding": "10px",
                                },
                            ),
                        ),
                        dbc.Input(
                            id="user-input",
                            placeholder="Ask something about your documents…",
                            type="text",
                            debounce=True,
                        ),
                        dbc.Button("Send", id="send-btn", color="primary", className="mt-2"),
                    ],
                    width=9,
                ),
            ],
            className="mt-4",
        ),
    ],
    fluid=True,
)

# ---------- 6. Callbacks ----------

# 6.1. Save API key to env (client side can't access env directly)
@app.callback(
    Output("api-key-input", "value", allow_duplicate=True),
    Input("api-key-input", "value"),
    prevent_initial_call=True,
)
def update_api_key(key):
    if key:
        os.environ["OPENAI_API_KEY"] = key.strip()
    return key

# 6.2. Handle file upload -> build retriever
@app.callback(
    Output("upload-status", "children"),
    Output("retriever-store", "data"),
    Output("qa-ready", "data"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    State("api-key-input", "value"),
    prevent_initial_call=True,
)
def handle_upload(contents, filenames, api_key):
    if not contents:
        return dash.no_update, dash.no_update, False

    if not api_key:
        return "❌ Please enter your OpenAI API key first.", dash.no_update, False
    os.environ["OPENAI_API_KEY"] = api_key.strip()

    uploaded_files = list(zip(contents, filenames))
    retriever = index_files(uploaded_files)

    sid = get_session_id()
    SESSION_STATE.setdefault(sid, {})
    SESSION_STATE[sid]["retriever"] = retriever
    SESSION_STATE[sid]["memory"] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    SESSION_STATE[sid]["qa_chain"] = None

    return "✅ Documents indexed. Ask away!", "ready", True

# 6.3. Clear chat
@app.callback(
    Output("chat-history", "data"),
    Output("chat-window", "children"),
    Input("clear-chat", "n_clicks"),
    prevent_initial_call=True,
)
def clear_chat(_):
    sid = get_session_id()
    if sid in SESSION_STATE:
        SESSION_STATE[sid]["memory"] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        SESSION_STATE[sid]["qa_chain"] = None
    return [], []

# 6.4. Send message
@app.callback(
    Output("chat-history", "data", allow_duplicate=True),
    Output("chat-window", "children", allow_duplicate=True),
    Input("send-btn", "n_clicks"),
    State("user-input", "value"),
    State("chat-history", "data"),
    State("model-select", "value"),
    State("temperature-slider", "value"),
    State("qa-ready", "data"),
    prevent_initial_call=True,
)
def send_message(_, user_text, history, model_name, temperature, qa_ready):
    if not user_text or not qa_ready:
        return dash.no_update, dash.no_update

    sid = get_session_id()
    state = SESSION_STATE.get(sid)
    if not state or "retriever" not in state:
        return dash.no_update, dash.no_update

    if state.get("qa_chain") is None:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        state["qa_chain"] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=state["retriever"],
            memory=state["memory"],
            combine_docs_chain_kwargs={
                "prompt": prompt_template,
                "document_variable_name": "context",
            },
        )

    history.append({"role": "user", "content": user_text})

    try:
        answer = state["qa_chain"].run(user_text)
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    history.append({"role": "assistant", "content": answer})

    chat_children = []
    for msg in history:
        bubble_color = "#e8f0fe" if msg["role"] == "user" else "#f1f3f4"
        align = "right" if msg["role"] == "user" else "left"
        chat_children.append(
            html.Div(
                msg["content"],
                style={
                    "background": bubble_color,
                    "padding": "8px 12px",
                    "borderRadius": "10px",
                    "margin": "5px 0",
                    "maxWidth": "80%",
                    "textAlign": "left",
                    "float": align,
                    "clear": "both",
                },
            )
        )

    return history, chat_children

# ---------- 7. Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)

