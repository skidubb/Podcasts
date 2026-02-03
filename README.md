# Podcast Semantic Search

A RAG (Retrieval-Augmented Generation) system for semantic search over podcast transcripts. Automatically transcribes episodes from any podcast RSS feed and provides AI-powered search with citations.

**See the main project in [`podcast-transcriber/`](./podcast-transcriber/) for full documentation and setup instructions.**

## Features

- **Automatic Transcription**: Syncs with any podcast RSS feed and transcribes using OpenAI Whisper
- **Semantic Search**: Find relevant content across all episodes using vector embeddings
- **AI-Powered Answers**: Get synthesized answers from Claude with source citations
- **Web Interface**: Clean Streamlit UI for searching
- **Cloud-Ready**: Deploy to Streamlit Cloud with Pinecone vector storage

## Quick Start

```bash
cd podcast-transcriber

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys and podcast RSS URL

# Transcribe episodes
python sync_episodes.py

# Build search index
python build_pinecone_index.py

# Run web app
streamlit run streamlit_app.py
```

## License

MIT
