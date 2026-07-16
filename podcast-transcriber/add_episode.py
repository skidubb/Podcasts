#!/usr/bin/env python3
"""Add a single episode to the existing Pinecone index.

This script adds one episode to the vector database without rebuilding the entire index.

Usage:
    python add_episode.py 93
    python add_episode.py 93 --dry-run  # Parse and chunk without uploading
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.config import Config
from rag.parser import PodcastParser
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.pinecone_indexer import PineconeIndexer


def find_episode_file(transcripts_dir: Path, episode_num: int) -> Path | None:
    """Find the transcript file for a given episode number."""
    # Look for files starting with the episode number (zero-padded to 3 digits)
    patterns = [
        f"{episode_num:03d}_*.md",
        f"{episode_num:02d}_*.md",
        f"{episode_num}_*.md",
    ]

    for pattern in patterns:
        matches = list(transcripts_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Add a single episode to the Pinecone index"
    )
    parser.add_argument(
        "episode_num",
        type=int,
        help="Episode number to add"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without uploading to Pinecone"
    )
    args = parser.parse_args()

    # Initialize config
    config = Config()

    # Validate configuration
    errors = config.validate()
    if not args.dry_run and not config.pinecone_api_key:
        errors.append("PINECONE_API_KEY not set in environment")
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(f"Transcripts directory: {config.transcripts_dir}")
    print(f"Pinecone index: {config.pinecone_index_name}")
    print(f"Episode to add: {args.episode_num}")
    print()

    # Find the episode file
    episode_file = find_episode_file(config.transcripts_dir, args.episode_num)
    if not episode_file:
        print(f"Error: Could not find transcript for episode {args.episode_num}")
        print(f"Looked in: {config.transcripts_dir}")
        sys.exit(1)

    print(f"Found transcript: {episode_file.name}")
    print()

    # Step 1: Parse the episode
    print("Step 1: Parsing transcript...")
    podcast_parser = PodcastParser(podcast_name=config.podcast_name)
    episode = podcast_parser.parse_file(episode_file)
    print(f"  Title: {episode.title}")
    print(f"  Guest: {episode.guest or 'N/A'}")
    print(f"  Word count: {episode.word_count:,}")
    print()

    # Step 2: Chunk the episode
    print("Step 2: Chunking transcript...")
    chunker = Chunker(
        chunk_size=config.chunk_size_tokens,
        chunk_overlap=config.chunk_overlap_tokens,
        min_chunk_size=config.min_chunk_tokens,
    )
    chunks = chunker.chunk_episode(episode)
    print(f"  Created {len(chunks)} chunks")

    total_tokens = sum(chunk.token_count for chunk in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per chunk: {avg_tokens:.1f}")
    print()

    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    embedder = Embedder(config)
    cost_estimate = embedder.estimate_cost(chunks)
    print(f"  Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")

    embeddings = embedder.embed_chunks(chunks, show_progress=True)
    print(f"  Generated {len(embeddings)} embeddings")
    print()

    if args.dry_run:
        print("Dry run complete. Skipping Pinecone upload.")
        print()
        print("Summary:")
        print(f"  Episode: {args.episode_num} - {episode.title}")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Embeddings: {len(embeddings)}")
        return

    # Step 4: Upsert to Pinecone
    print("Step 4: Uploading to Pinecone...")
    pinecone_indexer = PineconeIndexer(config)
    pinecone_indexer.create_index_if_not_exists()

    # Convert embeddings to list format
    embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
    total_upserted = pinecone_indexer.upsert_chunks(chunks, embeddings_list)
    print(f"  Uploaded {total_upserted} vectors")
    print()

    # Get stats
    stats = pinecone_indexer.get_stats()
    print("Pinecone Index Stats:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")
    print()

    print(f"Successfully added episode {args.episode_num} to the index!")


if __name__ == "__main__":
    main()
