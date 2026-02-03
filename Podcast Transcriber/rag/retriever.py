"""Retrieval with filtering and MMR diversity."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np

from .config import Config
from .chunker import Chunk
from .embedder import Embedder
from .indexer import FAISSIndexer

if TYPE_CHECKING:
    from .pinecone_indexer import PineconeIndexer


@dataclass
class RetrievalResult:
    """A retrieved chunk with score and metadata."""
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            'text': self.chunk.text,
            'score': self.score,
            'rank': self.rank,
            'episode_num': self.chunk.episode_num,
            'title': self.chunk.title,
            'guest': self.chunk.guest,
            'date': self.chunk.date,
            'citation': self.chunk.citation(),
        }


class Retriever:
    """Semantic search with filtering and diversity."""

    def __init__(
        self,
        config: Config,
        embedder: Embedder,
        indexer: FAISSIndexer,
    ):
        self.config = config
        self.embedder = embedder
        self.indexer = indexer

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        # Filters
        podcast: Optional[str] = None,
        episode_num: Optional[int] = None,
        guest: Optional[str] = None,
        date_from: Optional[str] = None,  # ISO format
        date_to: Optional[str] = None,
        # Diversity
        use_mmr: bool = True,
        mmr_lambda: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Search for relevant chunks with optional filtering and MMR diversity.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            podcast: Filter by podcast name
            episode_num: Filter by episode number
            guest: Filter by guest name (partial match)
            date_from: Filter by start date (ISO format)
            date_to: Filter by end date (ISO format)
            use_mmr: Whether to apply MMR for diversity
            mmr_lambda: MMR lambda (0=max diversity, 1=max relevance)
        """
        top_k = top_k or self.config.top_k
        min_score = min_score or self.config.similarity_threshold
        mmr_lambda = mmr_lambda if mmr_lambda is not None else self.config.mmr_lambda

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Get more results than needed for filtering
        fetch_k = min(top_k * 3, self.indexer.index.ntotal)
        raw_results = self.indexer.search(query_embedding, k=fetch_k)

        # Apply filters
        filtered = self._apply_filters(
            raw_results,
            podcast=podcast,
            episode_num=episode_num,
            guest=guest,
            date_from=date_from,
            date_to=date_to,
            min_score=min_score,
        )

        # Apply MMR for diversity
        if use_mmr and len(filtered) > top_k:
            filtered = self._apply_mmr(
                query_embedding,
                filtered,
                top_k=top_k,
                lambda_param=mmr_lambda,
            )
        else:
            filtered = filtered[:top_k]

        # Convert to RetrievalResult
        results = [
            RetrievalResult(chunk=chunk, score=score, rank=i + 1)
            for i, (chunk, score) in enumerate(filtered)
        ]

        return results

    def _apply_filters(
        self,
        results: list[tuple[Chunk, float]],
        podcast: Optional[str] = None,
        episode_num: Optional[int] = None,
        guest: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Apply metadata filters to results."""
        filtered = []

        for chunk, score in results:
            # Score threshold
            if score < min_score:
                continue

            # Podcast filter
            if podcast and chunk.podcast_name.lower() != podcast.lower():
                continue

            # Episode filter
            if episode_num is not None and chunk.episode_num != episode_num:
                continue

            # Guest filter (partial match, case-insensitive)
            if guest:
                if not chunk.guest or guest.lower() not in chunk.guest.lower():
                    continue

            # Date filters
            if chunk.date:
                chunk_date = chunk.date[:10]  # YYYY-MM-DD
                if date_from and chunk_date < date_from:
                    continue
                if date_to and chunk_date > date_to:
                    continue

            filtered.append((chunk, score))

        return filtered

    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        results: list[tuple[Chunk, float]],
        top_k: int,
        lambda_param: float,
    ) -> list[tuple[Chunk, float]]:
        """
        Apply Maximal Marginal Relevance for diversity.

        MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected))
        """
        if not results:
            return []

        # Get embeddings for candidates (re-embed the text)
        # Note: In production, you'd cache these
        candidate_texts = [chunk.text for chunk, _ in results]
        candidate_embeddings = np.array([
            self.embedder.embed_text(text) for text in candidate_texts
        ])

        selected_indices = []
        remaining_indices = list(range(len(results)))

        while len(selected_indices) < top_k and remaining_indices:
            best_idx = None
            best_mmr = float('-inf')

            for idx in remaining_indices:
                # Relevance to query
                relevance = results[idx][1]

                # Max similarity to already selected
                if selected_indices:
                    selected_embeddings = candidate_embeddings[selected_indices]
                    candidate_embedding = candidate_embeddings[idx:idx+1]

                    # Normalize for cosine similarity
                    selected_norm = selected_embeddings / np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
                    candidate_norm = candidate_embedding / np.linalg.norm(candidate_embedding, axis=1, keepdims=True)

                    similarities = np.dot(candidate_norm, selected_norm.T)
                    max_sim = float(np.max(similarities))
                else:
                    max_sim = 0.0

                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return [results[i] for i in selected_indices]

    def get_available_filters(self) -> dict:
        """Get available filter values from the index."""
        podcasts = set()
        guests = set()
        episodes = set()
        dates = []

        for chunk in self.indexer.chunks:
            podcasts.add(chunk.podcast_name)
            if chunk.guest:
                guests.add(chunk.guest)
            if chunk.episode_num:
                episodes.add(chunk.episode_num)
            if chunk.date:
                dates.append(chunk.date[:10])

        return {
            'podcasts': sorted(podcasts),
            'guests': sorted(guests),
            'episodes': sorted(episodes),
            'date_range': {
                'min': min(dates) if dates else None,
                'max': max(dates) if dates else None,
            },
        }


class PineconeRetriever:
    """Semantic search using Pinecone vector database."""

    def __init__(
        self,
        config: Config,
        embedder: Embedder,
        indexer: "PineconeIndexer",
    ):
        self.config = config
        self.embedder = embedder
        self.indexer = indexer

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        # Filters
        podcast: Optional[str] = None,
        episode_num: Optional[int] = None,
        guest: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        # Diversity (not implemented for Pinecone yet)
        use_mmr: bool = False,
        mmr_lambda: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Search for relevant chunks using Pinecone.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            podcast: Filter by podcast name
            episode_num: Filter by episode number
            guest: Filter by guest name (exact match in Pinecone)
            date_from: Filter by start date (ISO format)
            date_to: Filter by end date (ISO format)
            use_mmr: Not implemented for Pinecone
            mmr_lambda: Not implemented for Pinecone
        """
        top_k = top_k or self.config.top_k
        min_score = min_score or self.config.similarity_threshold

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Build Pinecone filter
        filter_dict = self._build_filter(
            podcast=podcast,
            episode_num=episode_num,
            guest=guest,
            date_from=date_from,
            date_to=date_to,
        )

        # Search Pinecone
        matches = self.indexer.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch extra for score filtering
            filter_dict=filter_dict if filter_dict else None,
        )

        # Convert to RetrievalResult
        results = []
        for i, match in enumerate(matches):
            score = match.score
            if score < min_score:
                continue

            chunk = self.indexer.chunk_from_match(match)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=i + 1))

            if len(results) >= top_k:
                break

        # Re-rank results
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _build_filter(
        self,
        podcast: Optional[str] = None,
        episode_num: Optional[int] = None,
        guest: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Optional[dict]:
        """Build Pinecone filter dictionary."""
        conditions = []

        if podcast:
            conditions.append({"podcast_name": {"$eq": podcast}})

        if episode_num is not None:
            conditions.append({"episode_num": {"$eq": episode_num}})

        if guest:
            # Pinecone doesn't support partial match, use exact match
            conditions.append({"guest": {"$eq": guest}})

        if date_from:
            conditions.append({"date": {"$gte": date_from}})

        if date_to:
            conditions.append({"date": {"$lte": date_to}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def get_available_filters(self) -> dict:
        """
        Get available filter values.

        Note: With Pinecone, we don't have local access to all chunks,
        so this returns empty/minimal values. The Insights tab should
        be disabled when using Pinecone.
        """
        return {
            'podcasts': [],
            'guests': [],
            'episodes': [],
            'date_range': {
                'min': None,
                'max': None,
            },
        }
