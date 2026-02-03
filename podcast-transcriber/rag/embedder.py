"""Embedding generation with caching and batching."""

import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

from .config import Config
from .chunker import Chunk


class Embedder:
    """Generate and cache embeddings using OpenAI API."""

    BATCH_SIZE = 100  # OpenAI recommends batches of ~100 for efficiency

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.embedding_model
        self.dimensions = config.embedding_dimensions
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path = config.embeddings_cache_path
        self._cache_index_path = config.data_dir / "embeddings_index.json"

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        self._cache[cache_key] = embedding
        return embedding

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True
    ) -> np.ndarray:
        """Embed multiple chunks with batching and progress reporting."""
        embeddings = []
        total = len(chunks)

        # Try to load from cache first
        cached_embeddings, missing_indices = self._load_cached_embeddings(chunks)
        if not missing_indices:
            if show_progress:
                print(f"Loaded all {total} embeddings from cache")
            return cached_embeddings

        if show_progress and cached_embeddings is not None:
            print(f"Found {total - len(missing_indices)} cached embeddings, generating {len(missing_indices)} new ones")

        # Initialize result array
        if cached_embeddings is not None:
            embeddings_array = cached_embeddings
        else:
            embeddings_array = np.zeros((total, self.dimensions), dtype=np.float32)

        # Generate missing embeddings in batches
        missing_chunks = [(i, chunks[i]) for i in missing_indices]
        for batch_start in range(0, len(missing_chunks), self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, len(missing_chunks))
            batch = missing_chunks[batch_start:batch_end]

            if show_progress:
                print(f"Embedding batch {batch_start // self.BATCH_SIZE + 1}/{(len(missing_chunks) + self.BATCH_SIZE - 1) // self.BATCH_SIZE}")

            texts = [chunk.text for _, chunk in batch]
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )

            for j, embedding_data in enumerate(response.data):
                original_idx = batch[j][0]
                embedding = np.array(embedding_data.embedding, dtype=np.float32)
                embeddings_array[original_idx] = embedding
                # Cache for future use
                self._cache[self._cache_key(texts[j])] = embedding

        # Save embeddings to cache
        self._save_embeddings_cache(chunks, embeddings_array)

        return embeddings_array

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query."""
        return self.embed_text(query)

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()

    def _load_cached_embeddings(
        self,
        chunks: list[Chunk]
    ) -> tuple[Optional[np.ndarray], list[int]]:
        """Load embeddings from cache, return missing indices."""
        if not self._cache_path.exists() or not self._cache_index_path.exists():
            return None, list(range(len(chunks)))

        # Load cache index
        with open(self._cache_index_path, 'r') as f:
            cache_index = json.load(f)

        # Check if cache is valid for these chunks
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if cache_index.get('chunk_ids') != chunk_ids:
            return None, list(range(len(chunks)))

        # Load embeddings
        embeddings = np.load(self._cache_path)

        # Check dimensions match
        if embeddings.shape != (len(chunks), self.dimensions):
            return None, list(range(len(chunks)))

        return embeddings, []

    def _save_embeddings_cache(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """Save embeddings to cache."""
        np.save(self._cache_path, embeddings)

        cache_index = {
            'chunk_ids': [chunk.chunk_id for chunk in chunks],
            'model': self.model,
            'dimensions': self.dimensions,
        }
        with open(self._cache_index_path, 'w') as f:
            json.dump(cache_index, f)

    def estimate_cost(self, chunks: list[Chunk]) -> dict:
        """Estimate embedding cost."""
        total_tokens = sum(chunk.token_count for chunk in chunks)
        # text-embedding-3-small: $0.02 per 1M tokens
        cost_per_million = 0.02
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million

        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'estimated_cost_usd': round(estimated_cost, 4),
            'model': self.model,
        }
