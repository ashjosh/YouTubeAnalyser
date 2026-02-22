# YouTube Transcript RAG (Standalone Streamlit App)

This is a **separate project** and does not modify the main Streamlit app in the repository root.

## Features
- Extract transcript from a YouTube video URL/ID (when captions are available)
- Token-aware transcript chunking using `tiktoken`
- Embedding generation with OpenAI embeddings
- FAISS vector index storage in memory
- Query answering over transcript chunks with source chunk IDs

## Run locally
```bash
cd youtube_transcript_rag
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Requires `OPENAI_API_KEY` or entering key in the app sidebar.
- Videos without captions cannot be indexed.
