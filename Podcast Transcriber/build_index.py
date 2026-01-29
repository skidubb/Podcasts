#!/usr/bin/env python3
"""Build the FAISS index from podcast transcripts."""

import sys
import time
from pathlib import Path

from rag.config import Config
from rag.parser import PodcastParser
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.indexer import FAISSIndexer


def main():
    print("=" * 60)
    print("Podcast RAG Index Builder")
    print("=" * 60)
    print()

    # Initialize config
    config = Config()

    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(f"Transcripts directory: {config.transcripts_dir}")
    print(f"Data directory: {config.data_dir}")
    print()

    # Step 1: Parse transcripts
    print("[1/4] Parsing transcripts...")
    start = time.time()

    parser = PodcastParser()
    episodes = parser.parse_directory(config.transcripts_dir)

    print(f"  Parsed {len(episodes)} episodes")
    total_words = sum(ep.word_count for ep in episodes)
    print(f"  Total words: {total_words:,}")
    print(f"  Time: {time.time() - start:.1f}s")
    print()

    # Step 2: Chunk episodes
    print("[2/4] Chunking episodes...")
    start = time.time()

    chunker = Chunker(
        chunk_size=config.chunk_size_tokens,
        chunk_overlap=config.chunk_overlap_tokens,
        min_chunk_size=config.min_chunk_tokens,
    )
    chunks = chunker.chunk_episodes(episodes)

    # Save chunks
    chunker.save_chunks(chunks, config.chunks_path)

    print(f"  Created {len(chunks)} chunks")
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
    print(f"  Average chunk size: {avg_tokens:.0f} tokens")
    print(f"  Saved to: {config.chunks_path}")
    print(f"  Time: {time.time() - start:.1f}s")
    print()

    # Step 3: Generate embeddings
    print("[3/4] Generating embeddings...")
    start = time.time()

    embedder = Embedder(config)

    # Show cost estimate
    cost = embedder.estimate_cost(chunks)
    print(f"  Estimated cost: ${cost['estimated_cost_usd']:.4f}")
    print(f"  Total tokens: {cost['total_tokens']:,}")

    embeddings = embedder.embed_chunks(chunks, show_progress=True)

    print(f"  Generated {len(embeddings)} embeddings")
    print(f"  Time: {time.time() - start:.1f}s")
    print()

    # Step 4: Build and save index
    print("[4/4] Building FAISS index...")
    start = time.time()

    indexer = FAISSIndexer(config)
    indexer.build_index(chunks, embeddings)
    indexer.save()

    stats = indexer.get_stats()
    print(f"  Index size: {stats['index_size_mb']:.1f} MB")
    print(f"  Time: {time.time() - start:.1f}s")
    print()

    # Summary
    print("=" * 60)
    print("Index built successfully!")
    print("=" * 60)
    print()
    print("Stats:")
    print(f"  Episodes: {stats['unique_episodes']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Unique guests: {stats['unique_guests']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Estimated content: {stats['estimated_hours']} hours")
    print()
    print("Files created:")
    print(f"  {config.faiss_index_path}")
    print(f"  {config.metadata_path}")
    print(f"  {config.chunks_path}")
    print()
    print("Run 'python cli.py chat' to start querying!")


if __name__ == "__main__":
    main()
