"""Configuration for RAG system."""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """Get a secret from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fall back to environment variables
    return os.getenv(key, default)


def slugify_index_name(name: str) -> str:
    """Convert podcast name to valid Pinecone index name.

    Pinecone index names: lowercase alphanumeric and hyphens, max 45 chars.
    """
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug[:45].strip('-')


@dataclass
class Config:
    """Configuration settings for the RAG system."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default=None)
    transcripts_dir: Path = field(default=None)

    # Podcast configuration
    rss_feed_url: str = field(default_factory=lambda: get_secret("RSS_FEED_URL"))
    podcast_name: str = field(default_factory=lambda: get_secret("PODCAST_NAME", "Podcast"))
    podcast_description: str = field(default_factory=lambda: get_secret(
        "PODCAST_DESCRIPTION",
        "A podcast featuring conversations with industry experts"
    ))

    # OpenAI settings (for embeddings)
    openai_api_key: str = field(default_factory=lambda: get_secret("OPENAI_API_KEY"))
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Anthropic settings (for generation)
    anthropic_api_key: str = field(default_factory=lambda: get_secret("ANTHROPIC_API_KEY"))
    llm_model: str = field(default_factory=lambda: get_secret("LLM_MODEL", "claude-sonnet-5"))

    # Pinecone settings (for cloud vector storage)
    pinecone_api_key: str = field(default_factory=lambda: get_secret("PINECONE_API_KEY"))
    pinecone_index_name: str = field(default=None)
    pinecone_region: str = "us-east-1"

    # Whisper settings
    whisper_model: str = field(default_factory=lambda: get_secret("WHISPER_MODEL", "base"))

    # Chunking settings
    chunk_size_tokens: int = 500  # Target 400-600 tokens
    chunk_overlap_tokens: int = 75  # 50-100 token overlap
    min_chunk_tokens: int = 50

    # Retrieval settings
    top_k: int = 10
    similarity_threshold: float = 0.1  # Lower threshold to ensure results
    mmr_lambda: float = 0.7  # Balance relevance vs diversity

    # Generation settings
    max_context_tokens: int = 6000
    temperature: float = 0.3

    def __post_init__(self):
        # Set Pinecone index name from env or derive from podcast name
        if self.pinecone_index_name is None:
            env_index = get_secret("PINECONE_INDEX_NAME")
            if env_index:
                self.pinecone_index_name = env_index
            else:
                self.pinecone_index_name = slugify_index_name(self.podcast_name)

        # Set data directory
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"

        # Set transcripts directory from env or default
        if self.transcripts_dir is None:
            env_transcripts = get_secret("TRANSCRIPTS_DIR")
            if env_transcripts:
                self.transcripts_dir = Path(env_transcripts)
                if not self.transcripts_dir.is_absolute():
                    self.transcripts_dir = self.base_dir / self.transcripts_dir
            else:
                # Default to ./transcripts
                self.transcripts_dir = self.base_dir / "transcripts"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "faiss_index").mkdir(exist_ok=True)

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss_index" / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "faiss_index" / "metadata.json"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def embeddings_cache_path(self) -> Path:
        return self.data_dir / "embeddings_cache.npy"

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY not set in environment")

        if not self.rss_feed_url:
            errors.append("RSS_FEED_URL not set in environment")

        if not self.transcripts_dir.exists():
            errors.append(f"Transcripts directory not found: {self.transcripts_dir}")

        return errors
