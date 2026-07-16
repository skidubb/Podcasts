# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system for semantic search over podcast transcripts. Transcribes episodes from any RSS feed using Whisper, indexes them in Pinecone, and answers questions using Claude with citations.

## Architecture

```
RSS Feed → sync_episodes.py (Whisper) → Markdown Transcripts
Transcripts → build_pinecone_index.py → Pinecone (parse → chunk → embed → upsert)
User Query → streamlit_app.py → Embed → Pinecone Search → Claude → Answer + Citations
```

All application code lives in `podcast-transcriber/`. The `rag/` package contains the core pipeline modules: config, parser, chunker, embedder, pinecone_indexer, retriever, generator.

## Commands

All commands run from `podcast-transcriber/`:

```bash
# Setup
cd podcast-transcriber
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys

# Sync episodes from RSS feed
python sync_episodes.py --dry-run           # preview
python sync_episodes.py                     # transcribe new episodes
python sync_episodes.py --rebuild-index     # also rebuild Pinecone index
python sync_episodes.py --force-episode 42  # re-transcribe specific episode
python sync_episodes.py --model medium      # use larger Whisper model

# Build/rebuild Pinecone index
python build_pinecone_index.py --dry-run    # preview
python build_pinecone_index.py              # build
python build_pinecone_index.py --rebuild    # delete and recreate

# Run web app
streamlit run streamlit_app.py              # http://localhost:8501

# CLI search
python cli.py "your question here"
```

There are no tests in this project.

## Key Configuration

Configuration is managed via `rag/config.py` which reads from `.env` files or Streamlit secrets (for cloud deployment). Key defaults:

- **LLM model**: `claude-opus-4-5-20251101` (set in config.py)
- **Embedding model**: `text-embedding-3-small` (1536 dimensions)
- **Chunk size**: 500 tokens, 75 token overlap, 50 token minimum
- **Retrieval**: top_k=10, similarity threshold=0.1, MMR lambda=0.7
- **Generation**: max 6000 context tokens, temperature=0.3
- **Pinecone**: serverless, us-east-1

## Dual Vector DB Support

- **Pinecone** (`pinecone_indexer.py`, `PineconeRetriever`): Primary, used for cloud deployment
- **FAISS** (`indexer.py`, `Retriever`): Legacy local-only option, still functional

## Deployment

- **Local**: `streamlit run streamlit_app.py`
- **Streamlit Cloud**: Set main file to `podcast-transcriber/streamlit_app.py`, add secrets in Streamlit UI
- **GitHub Actions**: Weekly sync workflow (`.github/workflows/sync-episodes.yml`) transcribes new episodes and rebuilds the index automatically. Requires GitHub Secrets for all API keys and podcast config.

## System Dependencies

- Python 3.11+
- ffmpeg (required for Whisper audio processing; `brew install ffmpeg` on Mac)
