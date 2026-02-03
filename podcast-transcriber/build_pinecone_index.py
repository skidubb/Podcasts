#!/usr/bin/env python3
"""Build Pinecone index from podcast transcripts.

This script:
1. Parses all markdown transcripts from the configured transcripts directory
2. Chunks them using token-based chunking (500 tokens, 75 overlap)
3. Generates embeddings using OpenAI text-embedding-3-small
4. Uploads vectors to Pinecone serverless index

Usage:
    python build_pinecone_index.py
    python build_pinecone_index.py --rebuild  # Delete and recreate index
    python build_pinecone_index.py --dry-run  # Parse and chunk without uploading
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


def main():
    parser = argparse.ArgumentParser(description="Build Pinecone index from podcast transcripts")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch"
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
    if not config.pinecone_api_key:
        errors.append("PINECONE_API_KEY not set in environment")
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(f"Transcripts directory: {config.transcripts_dir}")
    print(f"Pinecone index: {config.pinecone_index_name}")
    print()

    # Parse transcripts
    print("Step 1: Parsing transcripts...")
    podcast_parser = PodcastParser()
    episodes = podcast_parser.parse_directory(config.transcripts_dir)
    print(f"  Parsed {len(episodes)} episodes")

    total_words = sum(ep.word_count for ep in episodes)
    print(f"  Total words: {total_words:,}")
    print()

    # Chunk episodes
    print("Step 2: Chunking transcripts...")
    chunker = Chunker(
        chunk_size=config.chunk_size_tokens,
        chunk_overlap=config.chunk_overlap_tokens,
        min_chunk_size=config.min_chunk_tokens,
    )
    chunks = chunker.chunk_episodes(episodes)
    print(f"  Created {len(chunks)} chunks")

    total_tokens = sum(chunk.token_count for chunk in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per chunk: {avg_tokens:.1f}")
    print()

    # Generate embeddings
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
        print(f"  Episodes: {len(episodes)}")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Embeddings: {len(embeddings)}")
        return

    # Initialize Pinecone
    print("Step 4: Uploading to Pinecone...")
    pinecone_indexer = PineconeIndexer(config)

    # Create or check index
    pinecone_indexer.create_index_if_not_exists()

    # Optionally rebuild
    if args.rebuild:
        print("  Deleting existing vectors...")
        pinecone_indexer.delete_all()

    # Upsert chunks
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

    print("Index build complete!")
    print()
    print("Next steps:")
    print("  1. Verify in Pinecone console: https://app.pinecone.io")
    print("  2. Test locally: streamlit run streamlit_app.py")
    print("  3. Deploy to Streamlit Cloud for a shareable URL")


if __name__ == "__main__":
    main()
