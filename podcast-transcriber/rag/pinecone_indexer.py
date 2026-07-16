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
                # Entity fields are omitted (not written as empty lists) when
                # unset, so entity filters never match un-enriched chunks.
                for key in ("people", "companies", "products", "topics"):
                    values = getattr(chunk, key, None)
                    if values:
                        metadata[key] = values

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

    def update_episode_entities(
        self,
        podcast_name: str,
        episode_num: int,
        entities: dict,
        namespace: str = "",
    ) -> int:
        """Set entity metadata on every existing chunk of an episode.

        Partial metadata update only — vector values are untouched, so no
        re-embedding is needed. Returns the number of records updated.
        """
        # Empty lists are omitted — Pinecone silently ignores them anyway,
        # and un-enriched fields should stay invisible to entity filters.
        set_metadata = {
            key: entities[key]
            for key in ("people", "companies", "products", "topics")
            if entities.get(key)
        }
        if entities.get("guest"):
            set_metadata["guest"] = entities["guest"]
        if not set_metadata:
            return 0

        # Enumerate the episode's chunk IDs with a filtered query anchored on
        # chunk 0. GET-based fetch/list mangle IDs containing spaces, so stick
        # to POST-based query/update.
        first_id = f"{podcast_name}_{episode_num}_0"
        response = self.index.query(
            id=first_id,
            top_k=1000,
            filter={"episode_num": episode_num},
            include_metadata=False,
            namespace=namespace,
        )
        vector_ids = [match.id for match in response.matches]
        if not vector_ids:
            return 0

        updated = 0
        for vector_id in vector_ids:
            retry_with_backoff(
                self.index.update,
                id=vector_id,
                set_metadata=set_metadata,
                namespace=namespace,
                should_retry=is_retryable_status,
                label="Pinecone metadata update",
            )
            updated += 1
        return updated

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
            people=list(metadata.get("people", [])),
            companies=list(metadata.get("companies", [])),
            products=list(metadata.get("products", [])),
            topics=list(metadata.get("topics", [])),
        )
