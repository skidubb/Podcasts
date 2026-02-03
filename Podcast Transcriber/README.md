# Podcast Semantic Search

A RAG (Retrieval-Augmented Generation) system for semantic search over podcast transcripts. Automatically transcribes episodes from any podcast RSS feed and lets you ask questions using AI-powered search with citations.

## Features

- **Automatic Transcription**: Syncs with any podcast RSS feed and transcribes episodes using OpenAI Whisper
- **Semantic Search**: Find relevant content across all episodes using vector embeddings
- **AI-Powered Answers**: Get synthesized answers from Claude with source citations
- **Web Interface**: Clean Streamlit UI for searching and exploring
- **Cloud-Ready**: Deploy to Streamlit Cloud with Pinecone vector storage

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/podcast-semantic-search.git
cd podcast-semantic-search

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example config
cp .env.example .env

# Edit .env with your settings
```

Required settings in `.env`:

```bash
# API Keys
OPENAI_API_KEY=sk-your-key          # For embeddings
ANTHROPIC_API_KEY=sk-ant-your-key   # For Claude answers
PINECONE_API_KEY=pcsk_your-key      # For vector storage

# Your Podcast
RSS_FEED_URL=https://your-podcast-rss-feed.com/feed.xml
PODCAST_NAME=Your Podcast Name
PODCAST_DESCRIPTION=A brief description of your podcast content
```

### 3. Sync Episodes

```bash
# See what episodes will be transcribed
python sync_episodes.py --dry-run

# Transcribe all episodes (this takes time!)
python sync_episodes.py
```

### 4. Build Search Index

```bash
# Create Pinecone index and upload embeddings
python build_pinecone_index.py
```

### 5. Run the App

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vectors |
| `RSS_FEED_URL` | Yes | Your podcast's RSS feed URL |
| `PODCAST_NAME` | Yes | Display name for your podcast |
| `PODCAST_DESCRIPTION` | Yes | Brief description (used in AI prompts) |
| `PINECONE_INDEX_NAME` | No | Custom index name (default: slugified podcast name) |
| `TRANSCRIPTS_DIR` | No | Where to store transcripts (default: `./transcripts`) |
| `WHISPER_MODEL` | No | Whisper model: tiny, base, small, medium, large (default: base) |

## Sync Commands

```bash
# Check for new episodes (dry run)
python sync_episodes.py --dry-run

# Transcribe missing episodes
python sync_episodes.py

# Force re-transcribe a specific episode
python sync_episodes.py --force-episode 42

# Use a different Whisper model
python sync_episodes.py --model medium

# Transcribe and rebuild index
python sync_episodes.py --rebuild-index
```

## Deploy to Streamlit Cloud

1. Push your repo to GitHub (without transcripts or .env)

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub repo

4. Add secrets in Streamlit Cloud settings:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   PINECONE_API_KEY = "pcsk_..."
   PODCAST_NAME = "Your Podcast Name"
   PODCAST_DESCRIPTION = "Your podcast description"
   ```

5. Deploy!

## Project Structure

```
├── streamlit_app.py      # Web interface
├── sync_episodes.py      # RSS sync and transcription
├── build_pinecone_index.py  # Index builder
├── rag/
│   ├── config.py         # Configuration
│   ├── chunker.py        # Text chunking
│   ├── embedder.py       # OpenAI embeddings
│   ├── pinecone_indexer.py  # Pinecone operations
│   ├── retriever.py      # Search logic
│   └── generator.py      # Claude answer generation
├── .env.example          # Environment template
├── .gitignore           # Git exclusions
└── requirements.txt      # Python dependencies
```

## How It Works

```
RSS Feed → Whisper Transcription → Markdown Files
                                        ↓
                              Chunking (500 tokens)
                                        ↓
User Question → OpenAI Embed → Pinecone Search → Top K Chunks
                                                      ↓
                                              Claude Synthesis
                                                      ↓
                                              Answer + Citations
```

## Tips

- **Whisper Models**: Start with `base` for speed, use `medium` or `large` for better accuracy
- **Chunking**: Default 500 tokens with 75 token overlap works well for most podcasts
- **Top K**: Start with 10 results; increase if answers seem incomplete
- **Rate Limits**: The sync script includes rate limiting to avoid API issues

## License

MIT
