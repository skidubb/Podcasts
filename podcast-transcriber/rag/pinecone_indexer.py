"""Pinecone vector database indexer for podcast semantic search."""

import os
from typing import Optional

from pinecone import Pinecone, ServerlessSpec

from .config import Config
from .chunker import Chunk
from .retry import retry_with_backoff, is_retryable_status


class PineconeIndexer:
    """Pinecone client wrapper for vector storage and search."""

    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.pinecone_api_key
        self.index_name = config.pinecone_index_name
        self.dimension = config.embedding_dimensions

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")

        self.pc = Pinecone(api_key=self.api_key)
        self._index = None

    @property
    def index(self):
        """Lazy load the index."""
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def create_index_if_not_exists(self) -> bool:
        """
        Create a serverless Pinecone index if it doesn't exist.

        Returns:
            True if index was created, False if it already existed.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name in existing_indexes:
            print(f"Index '{self.index_name}' already exists")
            return False

        print(f"Creating index '{self.index_name}'...")
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{self.index_name}' created successfully")
        return True

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
        namespace: str = "",
    ) -> int:
        """
        Batch upload vectors with metadata to Pinecone.

        Args:
            chunks: List of Chunk objects with metadata
            embeddings: List of embedding vectors
            batch_size: Number of vectors per batch
            namespace: Optional namespace for organization

        Returns:
            Total number of vectors upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")

        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            vectors = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                # Convert numpy array to list if needed
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()

                metadata = {
                    "text": chunk.text[:8000],  # Pinecone metadata limit
                    "episode_num": chunk.episode_num or 0,
                    "title": chunk.title,
                    "guest": chunk.guest or "",
                    "date": chunk.date or "",
                    "podcast_name": chunk.podcast_name,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "token_count": chunk.token_count,
                }

                vectors.append({
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": metadata,
                })

            # Per-batch retry doubles as resume: a transient failure re-sends
            # only this batch, and upserts of already-sent ids are idempotent.
            retry_with_backoff(
                self.index.upsert,
                vectors=vectors,
                namespace=namespace,
                should_retry=is_retryable_status,
                label="Pinecone upsert",
            )
            total_upserted += len(vectors)
            print(f"Upserted batch {i // batch_size + 1}: {total_upserted}/{len(chunks)} vectors")

        return total_upserted

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[dict] = None,
        include_metadata: bool = True,
    ) -> list[dict]:
        """
        Query Pinecone for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            namespace: Optional namespace to search within
            filter_dict: Optional metadata filters
            include_metadata: Whether to include metadata in results

        Returns:
            List of match dictionaries with id, score, and metadata
        """
        # Convert numpy array to list if needed
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata,
        )

        return results.matches

    def delete_all(self, namespace: str = "") -> None:
        """Delete all vectors in a namespace."""
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted all vectors in namespace '{namespace or 'default'}'")

    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
        }

    def chunk_from_match(self, match: dict) -> Chunk:
        """Convert a Pinecone match result back to a Chunk object."""
        metadata = match.metadata
        return Chunk(
            chunk_id=match.id,
            text=metadata.get("text", ""),
            token_count=metadata.get("token_count", 0),
            episode_num=metadata.get("episode_num") or None,
            title=metadata.get("title", ""),
            guest=metadata.get("guest") or None,
            date=metadata.get("date") or None,
            podcast_name=metadata.get("podcast_name", ""),
            chunk_index=metadata.get("chunk_index", 0),
            total_chunks=metadata.get("total_chunks", 1),
        )
