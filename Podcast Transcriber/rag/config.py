"""Configuration for RAG system."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG system."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default=None)
    transcripts_dir: Path = field(default=None)

    # OpenAI settings (for embeddings)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Anthropic settings (for generation)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llm_model: str = "claude-opus-4-5-20251101"

    # Pinecone settings (for cloud vector storage)
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index_name: str = "gtm-ai-podcast"
    pinecone_region: str = "us-east-1"

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
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.transcripts_dir is None:
            # Check multiple possible locations for transcripts
            possible_paths = [
                self.base_dir.parent / "podcast_transcripts" / "GTM_AI_Podcast",
                Path("/Users/scottewalt/Documents/Podcasts/podcast_transcripts/GTM_AI_Podcast"),
                Path.home() / "Documents" / "Podcasts" / "podcast_transcripts" / "GTM_AI_Podcast",
            ]
            for path in possible_paths:
                if path.exists():
                    self.transcripts_dir = path
                    break
            else:
                # Default fallback
                self.transcripts_dir = possible_paths[0]

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

        if not self.transcripts_dir.exists():
            errors.append(f"Transcripts directory not found: {self.transcripts_dir}")

        return errors
