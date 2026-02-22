import json
import importlib
import os
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
import streamlit as st
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOKEN_LIMIT_PER_CHUNK = 500
TOKEN_OVERLAP = 60
VECTOR_DB_ROOT = Path("vector_store")


# -------------------------
# Session State
# -------------------------
def init_state():
    defaults = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "transcript_text": "",
        "chunks": [],
        "vector_store": None,
        "video_id": "",
        "vector_db_path": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -------------------------
# Video ID Extraction
# -------------------------
def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    parsed = urlparse(value)

    if parsed.netloc == "youtu.be":
        return parsed.path.strip("/")[:11]

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0][:11]
        if parsed.path.startswith(("/shorts/", "/embed/")):
            return parsed.path.split("/")[2][:11]

    return ""


# -------------------------
# Transcript Fetchers
# -------------------------
def _fetch_transcript_via_library(video_id: str) -> str:
    mod_spec = importlib.util.find_spec("youtube_transcript_api")
    if mod_spec is None:
        raise ValueError("youtube-transcript-api not installed")

    mod = importlib.import_module("youtube_transcript_api")
    api_cls = mod.YouTubeTranscriptApi

    try:
        # NEWER versions
        if hasattr(api_cls, "get_transcript"):
            transcript = api_cls.get_transcript(video_id, languages=["en", "hi"])

        # OLDER versions
        else:
            api = api_cls()
            transcript = api.fetch(video_id, languages=["en", "hi"])

    except Exception as e:
        raise ValueError(f"Library transcript fetch failed: {e}")

    parts = []

    for item in transcript:
        if isinstance(item, dict):
            text = item.get("text", "").strip()
        else:
            text = getattr(item, "text", "").strip()

        if text:
            parts.append(text)

    if not parts:
        raise ValueError("Transcript empty via library")

    return " ".join(parts)


def _fetch_transcript_via_timedtext(video_id: str) -> str:
    urls = [
        f"https://www.youtube.com/api/timedtext?lang=en&v={video_id}",
        f"https://www.youtube.com/api/timedtext?lang=en&kind=asr&v={video_id}",
    ]

    for url in urls:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            continue

        body = resp.text.strip()
        parts = []

        if body.startswith("<"):
            root = ET.fromstring(body)
            for node in root.findall(".//text"):
                text = unescape("".join(node.itertext())).strip()
                if text:
                    parts.append(text)

        if body.startswith("{"):
            events = resp.json().get("events", [])
            for event in events:
                for seg in event.get("segs", []):
                    text = seg.get("utf8", "").strip()
                    if text:
                        parts.append(text)

        if parts:
            return " ".join(parts)

    raise ValueError("No captions via timedtext endpoint")


def fetch_transcript(video_id: str) -> str:
    errors = []

    for fn in (_fetch_transcript_via_library, _fetch_transcript_via_timedtext):
        try:
            return fn(video_id)
        except Exception as exc:
            errors.append(str(exc))

    st.error("Transcript fetch failed. Reasons:")
    for err in errors:
        st.write("•", err)

    raise ValueError("Transcript unavailable")


# -------------------------
# Chunking
# -------------------------
def chunk_transcript_by_tokens(text: str, max_tokens: int, overlap: int):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    cid = 1

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        slice_tokens = tokens[start:end]

        chunks.append(
            {
                "chunk_id": cid,
                "text": enc.decode(slice_tokens),
                "token_count": len(slice_tokens),
                "start_token": start,
                "end_token": end,
            }
        )

        if end == len(tokens):
            break

        start = end - overlap
        cid += 1

    return chunks


# -------------------------
# Vector Store
# -------------------------
def build_vector_store(chunks, api_key: str):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    texts = [c["text"] for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)


def get_vector_db_path(video_id: str) -> Path:
    return VECTOR_DB_ROOT / video_id


def save_vector_store(vector_store: FAISS, video_id: str) -> Path:
    path = get_vector_db_path(video_id)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))
    return path


def load_vector_store(video_id: str, api_key: str):
    path = get_vector_db_path(video_id)
    if not path.exists():
        return None

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


# -------------------------
# QA
# -------------------------
def answer_query(query: str, vector_store, api_key: str) -> str:
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=api_key, temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant transcript chunks found."

    context = "\n\n".join(
        f"[chunk_id={d.metadata.get('chunk_id')}]\n{d.page_content}" for d in docs
    )

    prompt = f"""
Use ONLY this transcript context.

Context:
{context}

Question: {query}

Answer in bullet points and cite chunk IDs.
"""

    return llm.invoke(prompt).content


# -------------------------
# Actions
# -------------------------
def handle_index_actions(video_input: str, max_tokens: int, overlap: int):
    col1, col2 = st.columns(2)

    with col1:
        build_clicked = st.button("1) Build Index", use_container_width=True)

    with col2:
        load_clicked = st.button("Load Index", use_container_width=True)

    if build_clicked:
        if not st.session_state.openai_api_key:
            st.error("Provide OpenAI API Key")
            return

        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("Invalid video URL / ID")
            return

        try:
            with st.spinner("Fetching transcript..."):
                transcript = fetch_transcript(video_id)

            chunks = chunk_transcript_by_tokens(transcript, max_tokens, overlap)
            vs = build_vector_store(chunks, st.session_state.openai_api_key)
            path = save_vector_store(vs, video_id)

            st.session_state.video_id = video_id
            st.session_state.transcript_text = transcript
            st.session_state.chunks = chunks
            st.session_state.vector_store = vs
            st.session_state.vector_db_path = str(path)

            st.success("Index built successfully")

        except Exception as e:
            st.error(str(e))

    if load_clicked:
        video_id = extract_video_id(video_input)
        vs = load_vector_store(video_id, st.session_state.openai_api_key)

        if vs is None:
            st.error("Index not found. Build it first.")
            return

        st.session_state.vector_store = vs
        st.success("Index loaded")


# -------------------------
# UI
# -------------------------
def main():
    st.set_page_config(page_title="YouTube Transcript RAG", layout="wide")
    init_state()

    st.title("🎬 YouTube Transcript Q&A (RAG)")

    with st.sidebar:
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")

        max_tokens = st.slider("Tokens per chunk", 200, 1200, TOKEN_LIMIT_PER_CHUNK)
        overlap = st.slider("Overlap", 0, max_tokens - 1, min(TOKEN_OVERLAP, max_tokens - 1))

    video_input = st.text_input("YouTube URL or Video ID")

    handle_index_actions(video_input, max_tokens, overlap)

    st.markdown("---")

    query = st.text_input("Ask Question")

    if st.button("2) Answer"):
        if st.session_state.vector_store is None:
            st.error("Build or load index first")
            return

        with st.spinner("Generating answer..."):
            answer = answer_query(
                query,
                st.session_state.vector_store,
                st.session_state.openai_api_key,
            )

        st.write(answer)


if __name__ == "__main__":
    main()
