"""FAISS index management for vector similarity search."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from .config import Config
from .chunker import Chunk


class FAISSIndexer:
    """Manage FAISS index for semantic search."""

    def __init__(self, config: Config):
        self.config = config
        self.dimensions = config.embedding_dimensions
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[Chunk] = []

    def build_index(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Build FAISS index from chunks and embeddings."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings for {len(chunks)} chunks"
            )

        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Create flat index for exact search (fast enough for <100k vectors)
        self.index = faiss.IndexFlatIP(self.dimensions)
        self.index.add(embeddings)
        self.chunks = chunks

        print(f"Built index with {self.index.ntotal} vectors")

    def save(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save - build index first")

        index_path = index_path or self.config.faiss_index_path
        metadata_path = metadata_path or self.config.metadata_path

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save chunk metadata
        metadata = {
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'dimensions': self.dimensions,
            'total_vectors': self.index.ntotal,
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Saved index to {index_path}")
        print(f"Saved metadata to {metadata_path}")

    def load(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> bool:
        """Load index and metadata from disk. Returns True if successful."""
        index_path = index_path or self.config.faiss_index_path
        metadata_path = metadata_path or self.config.metadata_path

        if not index_path.exists() or not metadata_path.exists():
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load chunk metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.chunks = [Chunk.from_dict(c) for c in metadata['chunks']]

        print(f"Loaded index with {self.index.ntotal} vectors")
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Search index for similar chunks."""
        if self.index is None:
            raise ValueError("No index loaded - build or load index first")

        # Normalize query for cosine similarity
        query_normalized = query_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(query_normalized)

        # Search
        scores, indices = self.index.search(query_normalized, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                results.append((self.chunks[idx], float(score)))

        return results

    def get_stats(self) -> dict:
        """Get index statistics."""
        if self.index is None:
            return {'status': 'not_loaded'}

        # Collect unique values
        episodes = set()
        guests = set()
        dates = set()
        total_tokens = 0

        for chunk in self.chunks:
            if chunk.episode_num:
                episodes.add(chunk.episode_num)
            if chunk.guest:
                guests.add(chunk.guest)
            if chunk.date:
                dates.add(chunk.date[:7])  # Year-month
            total_tokens += chunk.token_count

        # Estimate hours of content (rough: 150 words/min speaking rate)
        total_words = sum(len(c.text.split()) for c in self.chunks)
        estimated_hours = total_words / 150 / 60

        return {
            'status': 'loaded',
            'total_vectors': self.index.ntotal,
            'total_chunks': len(self.chunks),
            'total_tokens': total_tokens,
            'unique_episodes': len(episodes),
            'unique_guests': len(guests),
            'date_range': f"{min(dates)} to {max(dates)}" if dates else None,
            'estimated_hours': round(estimated_hours, 1),
            'index_size_mb': round(self.index.ntotal * self.dimensions * 4 / 1_000_000, 2),
        }
