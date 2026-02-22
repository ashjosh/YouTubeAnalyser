import json
import importlib
import json
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


def init_state() -> None:
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "video_id" not in st.session_state:
        st.session_state.video_id = ""
    if "vector_db_path" not in st.session_state:
        st.session_state.vector_db_path = ""


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    parsed = urlparse(value)
    if parsed.netloc in {"youtu.be"}:
        return parsed.path.strip("/")[:11]

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0][:11]
        if parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[2][:11]

    return ""


def _fetch_transcript_via_library(video_id: str) -> str:
    """Try youtube-transcript-api when available in deployment environment."""
    mod_spec = importlib.util.find_spec("youtube_transcript_api")
    if mod_spec is None:
        raise ValueError("youtube-transcript-api package not installed.")

    mod = importlib.import_module("youtube_transcript_api")
    api = mod.YouTubeTranscriptApi
    api_cls = mod.YouTubeTranscriptApi

    transcript = None
    languages = ["en", "en-US", "en-GB", "hi"]

    if hasattr(api_cls, "get_transcript"):
        transcript = api_cls.get_transcript(video_id, languages=languages)
    else:
        api = api_cls()
        transcript = api.fetch(video_id, languages=languages)

    parts = []
    for item in transcript:
        if isinstance(item, dict):
            text = item.get("text", "").strip()
        else:
            text = getattr(item, "text", "").strip()
        if text:
            parts.append(text)

    transcript = api.get_transcript(video_id, languages=["en", "en-US", "en-GB", "hi"])
    parts = [item.get("text", "").strip() for item in transcript if item.get("text")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("Transcript fetched by library but empty.")
    return " ".join(parts)


def _fetch_transcript_via_timedtext(video_id: str) -> str:
    """Try YouTube timedtext endpoint directly (works for many captioned videos)."""
    candidates = [
        f"https://www.youtube.com/api/timedtext?lang=en&v={video_id}&fmt=json3",
        f"https://www.youtube.com/api/timedtext?lang=en&kind=asr&v={video_id}&fmt=json3",
        f"https://www.youtube.com/api/timedtext?lang=hi&v={video_id}&fmt=json3",
        f"https://www.youtube.com/api/timedtext?lang=hi&kind=asr&v={video_id}&fmt=json3",
    ]

    for url in candidates:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            continue
        payload = resp.json() if resp.text.strip().startswith("{") else {}
        events = payload.get("events", [])

        parts = []
        for event in events:
            for seg in event.get("segs", []):
                text = seg.get("utf8", "").strip()
        body = resp.text.strip()
        if body.startswith("{"):
            payload = resp.json()
            events = payload.get("events", [])
            for event in events:
                for seg in event.get("segs", []):
                    text = seg.get("utf8", "").strip()
                    if text:
                        parts.append(text)
        elif body.startswith("<"):
            root = ET.fromstring(resp.text)
            for node in root.findall('.//text'):
                text = unescape("".join(node.itertext())).strip()
                if text:
                    parts.append(text)

        if parts:
            return " ".join(parts)

    raise ValueError("No captions found via timedtext endpoint.")




def _extract_caption_tracks_from_watch_html(html: str):
    match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\});", html)
    if match:
        try:
            player_response = json.loads(match.group(1))
            captions = player_response.get("captions", {}).get("playerCaptionsTracklistRenderer", {})
            tracks = captions.get("captionTracks", [])
            if tracks:
                return tracks
        except json.JSONDecodeError:
            pass

    fallback_match = re.search(r'"captionTracks"\s*:\s*(\[.*?\])\s*,\s*"audioTracks"', html, flags=re.S)
    if fallback_match:
        try:
            return json.loads(fallback_match.group(1))
        except json.JSONDecodeError:
            return []
    return []

def _fetch_transcript_via_page_scrape(video_id: str) -> str:
    """Fallback transcript fetcher from watch page caption track metadata."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(
        watch_url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\});", response.text)
    if not match:
        raise ValueError("Could not read player response from YouTube page.")

    player_response = json.loads(match.group(1))
    captions = player_response.get("captions", {}).get("playerCaptionsTracklistRenderer", {})
    tracks = captions.get("captionTracks", [])
    tracks = _extract_caption_tracks_from_watch_html(response.text)
    if not tracks:
        raise ValueError("No captions/transcript available for this video.")

    selected_track = next((t for t in tracks if t.get("languageCode") == "en"), tracks[0])
    transcript_url = selected_track.get("baseUrl")
    if not transcript_url:
        raise ValueError("Caption URL missing from video metadata.")

    transcript_resp = requests.get(transcript_url + "&fmt=json3", timeout=20)
    transcript_resp.raise_for_status()

    events = transcript_resp.json().get("events", [])
    parts = []
    for event in events:
        for seg in event.get("segs", []):
            text = seg.get("utf8", "").strip()
    body = transcript_resp.text.strip()
    if body.startswith("{"):
        events = transcript_resp.json().get("events", [])
        for event in events:
            for seg in event.get("segs", []):
                text = seg.get("utf8", "").strip()
                if text:
                    parts.append(text)
    else:
        root = ET.fromstring(transcript_resp.text)
        for node in root.findall('.//text'):
            text = unescape("".join(node.itertext())).strip()
            if text:
                parts.append(text)

    if not parts:
        raise ValueError("Transcript fetched but empty.")

    return " ".join(parts)


def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript with a robust 2-step strategy:
    1) youtube-transcript-api
    2) page-scrape fallback
    """
    errors = []
    for fn in (_fetch_transcript_via_library, _fetch_transcript_via_timedtext, _fetch_transcript_via_page_scrape):
        try:
            return fn(video_id)
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        "No captions/transcript available for this video. "
        "Tried library + timedtext + page fallback. Details: " + " | ".join(errors)
    )


def chunk_transcript_by_tokens(text: str, max_tokens: int, overlap: int):
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0.")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens.")

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    chunk_id = 1
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        token_slice = tokens[start:end]
        chunk_text = enc.decode(token_slice)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(token_slice),
                "start_token": start,
                "end_token": end,
            }
        )
        if end == len(tokens):
            break
        start = end - overlap
        chunk_id += 1
    return chunks


def build_vector_store(chunks, api_key: str):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    texts = [item["text"] for item in chunks]
    metadatas = [
        {
            "chunk_id": item["chunk_id"],
            "token_count": item["token_count"],
            "start_token": item["start_token"],
            "end_token": item["end_token"],
        }
        for item in chunks
    ]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)


def get_vector_db_path(video_id: str) -> Path:
    return VECTOR_DB_ROOT / video_id


def save_vector_store(vector_store: FAISS, video_id: str) -> Path:
    path = get_vector_db_path(video_id)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))
    return path


def save_index_metadata(video_id: str, transcript_text: str, chunks) -> None:
    path = get_vector_db_path(video_id)
    metadata_path = path / "metadata.json"
    payload = {
        "video_id": video_id,
        "transcript_preview": transcript_text[:5000],
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_metadata(video_id: str):
    metadata_path = get_vector_db_path(video_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_vector_store(video_id: str, api_key: str):
    path = get_vector_db_path(video_id)
    if not path.exists():
        return None

    index_file = path / "index.faiss"
    store_file = path / "index.pkl"
    if not index_file.exists() or not store_file.exists():
        return None

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


def answer_query(query: str, vector_store, api_key: str) -> str:
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=api_key, temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    source_documents = retriever.invoke(query)
    if not source_documents:
        return "No relevant chunks were found in the vector index."

    context_blocks = []
    source_chunks = set()
    for doc in source_documents:
        cid = doc.metadata.get("chunk_id", "?")
        source_chunks.add(cid)
        context_blocks.append(f"[chunk_id={cid}]\n{doc.page_content}")

    context = "\n\n".join(context_blocks)
    prompt = (
        "You are helping users understand a YouTube video transcript.\n"
        "Use only the provided context chunks from the transcript.\n"
        "If the answer is not present, clearly say it is not found in the transcript.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer with short bullet points and mention chunk IDs when relevant."
    )

    answer = llm.invoke(prompt).content
    sorted_chunks = sorted(source_chunks, key=lambda x: (isinstance(x, str), x))
    return f"{answer}\n\nSources: chunk(s) {', '.join(map(str, sorted_chunks))}"


def main() -> None:
    st.set_page_config(page_title="YouTube Transcript RAG", layout="wide")
    init_state()

    st.title("🎬 YouTube Transcript Q&A (RAG)")
    st.write(
        "Extract a YouTube transcript, chunk it by token count, embed chunks, store in FAISS, and ask questions about the video."
    )

    with st.sidebar:
        st.header("Setup")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
        )
        st.caption("Your key is used only in this running session.")

        st.subheader("Chunking Controls")
        max_tokens = st.slider("Tokens per chunk", 200, 1200, TOKEN_LIMIT_PER_CHUNK, 50)
        overlap = st.slider("Token overlap", 0, 200, TOKEN_OVERLAP, 10)
        overlap_upper_bound = min(200, max_tokens - 1)
        overlap_default = min(TOKEN_OVERLAP, overlap_upper_bound)
        overlap = st.slider("Token overlap", 0, overlap_upper_bound, overlap_default, 10)

        st.subheader("Vector Database")
        st.caption("Indexes are persisted locally under `vector_store/<video_id>`.")
        if st.session_state.vector_db_path:
            st.code(st.session_state.vector_db_path)

    video_input = st.text_input("Paste YouTube URL or Video ID")

    if st.button("1) Extract Transcript and Build Index", use_container_width=True):
    col_actions_1, col_actions_2 = st.columns(2)
    with col_actions_1:
        build_clicked = st.button("1) Extract Transcript and Build Index", use_container_width=True)
    with col_actions_2:
        load_clicked = st.button("Load Existing Index", use_container_width=True)

    if build_clicked:
        if not st.session_state.openai_api_key:
            st.error("Please provide an OpenAI API key.")
            return

        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("Could not parse a valid YouTube video ID.")
            return

        try:
            with st.spinner("Fetching transcript and building embeddings..."):
                transcript_text = fetch_transcript(video_id)
                chunks = chunk_transcript_by_tokens(transcript_text, max_tokens=max_tokens, overlap=overlap)
                vector_store = build_vector_store(chunks, st.session_state.openai_api_key)
                db_path = save_vector_store(vector_store, video_id)
                save_index_metadata(video_id, transcript_text, chunks)

            st.session_state.video_id = video_id
            st.session_state.transcript_text = transcript_text
            st.session_state.chunks = chunks
            st.session_state.vector_store = vector_store
            st.session_state.vector_db_path = str(db_path)
            st.success(f"Indexed video `{video_id}` with {len(chunks)} chunk(s).")

        except Exception as exc:
            st.error(f"Failed to process video: {exc}")

    if load_clicked:
        if not st.session_state.openai_api_key:
            st.error("Please provide an OpenAI API key.")
            return

        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("Could not parse a valid YouTube video ID.")
            return

        try:
            vector_store = load_vector_store(video_id, st.session_state.openai_api_key)
        except Exception as exc:
            st.error(f"Failed to load saved index: {exc}")
            return

        if vector_store is None:
            st.error("No saved vector index found for this video ID. Build it first.")
        else:
            metadata = load_index_metadata(video_id)
            st.session_state.video_id = video_id
            st.session_state.vector_store = vector_store
            st.session_state.vector_db_path = str(get_vector_db_path(video_id))
            if metadata:
                st.session_state.transcript_text = metadata.get("transcript_preview", "")
                st.session_state.chunks = metadata.get("chunks", [])
            else:
                st.session_state.transcript_text = ""
                st.session_state.chunks = []
            st.success(f"Loaded saved FAISS index for `{video_id}`.")

    st.markdown("---")
    col1, col2 = st.columns([1.1, 1.4])

    with col1:
        st.subheader("Transcript + Chunk Stats")
        if st.session_state.chunks:
            total_tokens = sum(item["token_count"] for item in st.session_state.chunks)
            st.metric("Total Chunks", len(st.session_state.chunks))
            st.metric("Approx Total Tokens", total_tokens)
            st.dataframe(
                [
                    {
                        "chunk_id": item["chunk_id"],
                        "token_count": item["token_count"],
                        "token_range": f"{item['start_token']}-{item['end_token']}",
                    }
                    for item in st.session_state.chunks
                ],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Preview transcript"):
                st.write(st.session_state.transcript_text[:5000])
        else:
            st.info("No transcript indexed yet.")
            st.info("No transcript indexed yet. You can still load an existing saved index.")

    with col2:
        st.subheader("Ask Questions About the Video")
        query = st.text_input("Ask a question from this transcript")

        if st.button("2) Answer Query", use_container_width=True):
            if st.session_state.vector_store is None:
                st.error("Please extract/index a transcript first.")
                return
            if not query.strip():
                st.error("Please enter a question.")
                return

            with st.spinner("Generating answer from transcript chunks..."):
                response = answer_query(query, st.session_state.vector_store, st.session_state.openai_api_key)
            st.markdown(response)


if __name__ == "__main__":
    main()
