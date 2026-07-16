#!/usr/bin/env python3
"""Index Lenny's Podcast transcripts into Pinecone ce-gtm-knowledge index.

Uses Pinecone's integrated inference (llama-text-embed-v2) — upserts text
records directly, no OpenAI embeddings needed.
"""

import argparse
import importlib
import os
import re
import sys
import types
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

# Prevent rag/__init__.py from importing heavy deps (numpy, openai, etc.)
# by pre-registering the rag package as an empty module.
rag_pkg = types.ModuleType("rag")
rag_pkg.__path__ = [str(Path(__file__).parent / "rag")]
rag_pkg.__package__ = "rag"
sys.modules["rag"] = rag_pkg

from rag.parser import Episode  # noqa: E402
from rag.chunker import Chunker  # noqa: E402

load_dotenv()

TRANSCRIPTS_DIR = Path("/Users/scottewalt/Documents/Podcasts/podcast_transcripts/Lenny's Podcast")
INDEX_NAME = "ce-gtm-knowledge"
NAMESPACE = "lennys-podcast"
PODCAST_NAME = "Lenny's Podcast"
BATCH_SIZE = 20  # Records per upsert batch (small to stay under 250K tokens/min)


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')


def clean_transcript(text: str) -> str:
    """Clean Lenny's Podcast transcript: remove timestamps, normalize whitespace."""
    # Remove timestamps like "(00:01:21):" or "(HH:MM:SS):"
    text = re.sub(r'\((\d{1,2}:\d{2}:\d{2})\):\s*', '', text)
    # Remove standalone timestamps at line start
    text = re.sub(r'^\(\d{1,2}:\d{2}:\d{2}\)\s*', '', text, flags=re.MULTILINE)
    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_episodes(transcripts_dir: Path) -> list[Episode]:
    """Load all .txt transcripts as Episode objects."""
    episodes = []
    txt_files = sorted(transcripts_dir.glob("*.txt"))

    for file_path in txt_files:
        guest = file_path.stem  # filename minus .txt
        raw_text = file_path.read_text(encoding='utf-8')
        transcript = clean_transcript(raw_text)

        episode = Episode(
            episode_num=None,
            title=guest,
            guest=guest,
            date=None,
            duration_minutes=None,
            file_path=file_path,
            transcript=transcript,
            word_count=len(transcript.split()),
            podcast_name=PODCAST_NAME,
        )
        episodes.append(episode)

    return episodes


def build_records(episodes: list[Episode], chunker: Chunker) -> list[dict]:
    """Chunk episodes and build Pinecone records for integrated inference."""
    records = []
    for episode in episodes:
        chunks = chunker.chunk_episode(episode)
        guest_slug = slugify(episode.guest)
        for chunk in chunks:
            record = {
                "_id": f"lennys-podcast_{guest_slug}_{chunk.chunk_index}",
                "text": chunk.text,
                "guest": episode.guest,
                "podcast_name": PODCAST_NAME,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            }
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description="Index Lenny's Podcast transcripts into Pinecone")
    parser.add_argument("--dry-run", action="store_true", help="Preview without upserting")
    parser.add_argument("--start-batch", type=int, default=1, help="Resume from batch N (1-indexed)")
    args = parser.parse_args()

    print(f"Loading transcripts from {TRANSCRIPTS_DIR}")
    episodes = load_episodes(TRANSCRIPTS_DIR)
    print(f"Loaded {len(episodes)} episodes")

    chunker = Chunker(chunk_size=500, chunk_overlap=75, min_chunk_size=50)
    records = build_records(episodes, chunker)
    print(f"Created {len(records)} chunks from {len(episodes)} episodes")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        # Show a few sample records
        for r in records[:3]:
            print(f"\n  ID: {r['_id']}")
            print(f"  Guest: {r['guest']}")
            print(f"  Chunk: {r['chunk_index']}/{r['total_chunks']}")
            print(f"  Text: {r['text'][:100]}...")
        print(f"\nWould upsert {len(records)} records to {INDEX_NAME}/{NAMESPACE}")
        return

    # Connect to Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    # Batch upsert with retry on rate limits
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    start_offset = (args.start_batch - 1) * BATCH_SIZE
    if start_offset > 0:
        print(f"Resuming from batch {args.start_batch} (skipping first {start_offset} records)")
    for i in range(start_offset, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        for attempt in range(5):
            try:
                print(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} records)...")
                index.upsert_records(namespace=NAMESPACE, records=batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        # Delay between batches to stay under 250K tokens/min
        if batch_num < total_batches:
            time.sleep(3)

    print(f"\nDone! Upserted {len(records)} records to {INDEX_NAME}/{NAMESPACE}")


if __name__ == "__main__":
    main()
