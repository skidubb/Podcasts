# Podcast Semantic Search - Developer Guide

## Overview
RAG (Retrieval-Augmented Generation) system for semantic search over podcast transcripts. Works with any podcast RSS feed.

## Architecture
```
User Question → OpenAI Embed → Pinecone Search → Top K Chunks → Claude → Answer + Citations
```

## Tech Stack
- **Vector DB**: Pinecone (serverless)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dims)
- **LLM**: Claude (configurable: Sonnet or Opus)
- **Web UI**: Streamlit
- **Transcription**: OpenAI Whisper

## Key Files
| File | Purpose |
|------|---------|
| `streamlit_app.py` | Web interface |
| `sync_episodes.py` | RSS sync and transcription |
| `build_pinecone_index.py` | Index builder |
| `rag/config.py` | Configuration (reads from .env) |
| `rag/chunker.py` | Token-based text chunking |
| `rag/embedder.py` | OpenAI embedding generation |
| `rag/pinecone_indexer.py` | Pinecone vector operations |
| `rag/retriever.py` | Search logic |
| `rag/entity_vocab.py` | Entity vocabulary + variant expansion for filters |
| `rag/generator.py` | Claude answer generation |

## Configuration
All settings come from environment variables (`.env` file or Streamlit secrets):

| Variable | Required | Default |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | - |
| `ANTHROPIC_API_KEY` | Yes | - |
| `PINECONE_API_KEY` | Yes | - |
| `RSS_FEED_URL` | Yes | - |
| `PODCAST_NAME` | Yes | "Podcast" |
| `PODCAST_DESCRIPTION` | Yes | Generic description |
| `PINECONE_INDEX_NAME` | No | Slugified podcast name |
| `TRANSCRIPTS_DIR` | No | `./transcripts` |
| `WHISPER_MODEL` | No | `base` |

## Commands

### Sync Episodes
```bash
python sync_episodes.py --dry-run      # Preview
python sync_episodes.py                 # Transcribe missing episodes
python sync_episodes.py --model medium  # Use larger Whisper model
```

### Build Index
```bash
python build_pinecone_index.py --dry-run  # Preview
python build_pinecone_index.py            # Build index
python build_pinecone_index.py --rebuild  # Delete and rebuild
```

### Run Web App
```bash
streamlit run streamlit_app.py
# Access at http://localhost:8501
```

## Chunking Settings
- Chunk size: 500 tokens
- Overlap: 75 tokens
- Min chunk: 50 tokens

## Retrieval Settings
- Top K: 10 (configurable in UI)
- Similarity threshold: 0.1
- MMR lambda: 0.7 (balance relevance vs diversity)

## Entity Filters

Every chunk carries `people`, `companies`, `products`, and `topics` metadata
(see `rag/entity_extractor.py`). `Retriever.search()` and
`PineconeRetriever.search()` accept them as filters, plus `any_entity` to match
one value across all four:

```python
retriever.search(q, topics="patient acquisition", companies=["RealSelf"])
retriever.search(q, any_entity="Instagram")
```

Separate arguments are AND'd; multiple values in one argument are OR'd.

Extraction runs per episode with no shared vocabulary, so the same entity has
several surface forms ("ChatGPT" / "Chat GPT"). Filter values are therefore
expanded through `EntityVocabulary`, built from the committed
`entities/<slug>/*.json` caches, so case and punctuation don't matter. The
vocabulary is keyed on the **transcripts directory name**, not the slugified
podcast name.

Expansion only merges forms that are identical once case and punctuation are
stripped. It does NOT collapse transcription variants of one name
("Catherine Maley" / "Katherine Maley") — that needs a reviewed alias map,
since fuzzy matching would also merge distinct entities.
