# YouTube Transcript RAG (Standalone Streamlit App)
# YouTube Transcript RAG (Streamlit)

This is a **separate project** and does not modify the main Streamlit app in the repository root.
This app lets you run Q&A over any YouTube video transcript using a Retrieval-Augmented Generation (RAG) workflow.

## Features
- Extract transcript from a YouTube video URL/ID (when captions are available)
- Token-aware transcript chunking using `tiktoken`
- Embedding generation with OpenAI embeddings
- FAISS vector index storage in memory
- Query answering over transcript chunks with source chunk IDs
## What it does
- Extract transcript from a YouTube URL or video ID (when captions are available)
- Chunk transcript with token-aware splitting (`tiktoken`)
- Generate embeddings using OpenAI embeddings
- Store and load FAISS vector indexes locally per video (`vector_store/<video_id>`)
- Persist transcript/chunk metadata next to FAISS index for later inspection
- Answer user queries using retrieved transcript chunks with source chunk IDs

## Run locally
```bash
cd youtube_transcript_rag
pip install -r requirements.txt
streamlit run app.py
```

## Requirements
- OpenAI API key (`OPENAI_API_KEY`) or enter it in the app sidebar.
- Internet access for transcript retrieval and OpenAI API calls.
- Videos must have captions/transcripts available.
- Works with newer `youtube-transcript-api` releases that use `YouTubeTranscriptApi().fetch(...)`.

## Notes
- Requires `OPENAI_API_KEY` or entering key in the app sidebar.
- Videos without captions cannot be indexed.
- The vector index is saved locally so you can reuse it later via **Load Existing Index**.
- Loading validates required FAISS files (`index.faiss` and `index.pkl`) and shows a clear error if the saved index is incomplete/corrupt.
- Overlap is automatically constrained to always be less than chunk size to avoid invalid chunking settings.
- If no captions exist for a video, indexing will fail with an explanatory message.
app.py
app.py
