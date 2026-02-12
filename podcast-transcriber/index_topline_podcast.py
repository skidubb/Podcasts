#!/usr/bin/env python3
"""Index Topline podcast transcripts into Pinecone ce-gtm-knowledge index.

Uses Pinecone's integrated inference (llama-text-embed-v2) — upserts text
records directly, no OpenAI embeddings needed.
"""

import argparse
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

from rag.parser import PodcastParser  # noqa: E402
from rag.chunker import Chunker  # noqa: E402

load_dotenv()

TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts_topline"
INDEX_NAME = "ce-gtm-knowledge"
NAMESPACE = "topline-podcast"
PODCAST_NAME = "Topline"
BATCH_SIZE = 20


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')


def build_records(episodes, chunker):
    """Chunk episodes and build Pinecone records for integrated inference."""
    records = []
    for episode in episodes:
        chunks = chunker.chunk_episode(episode)
        title_slug = slugify(episode.title[:60])
        for chunk in chunks:
            record = {
                "_id": f"topline_{title_slug}_{chunk.chunk_index}",
                "text": chunk.text,
                "title": episode.title,
                "podcast_name": PODCAST_NAME,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            }
            if episode.guest:
                record["guest"] = episode.guest
            if episode.episode_num is not None:
                record["episode_num"] = episode.episode_num
            if episode.date:
                record["date"] = episode.date.isoformat()
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description="Index Topline podcast transcripts into Pinecone")
    parser.add_argument("--dry-run", action="store_true", help="Preview without upserting")
    parser.add_argument("--start-batch", type=int, default=1, help="Resume from batch N (1-indexed)")
    args = parser.parse_args()

    if not TRANSCRIPTS_DIR.exists():
        print(f"Transcripts directory not found: {TRANSCRIPTS_DIR}")
        print("Run sync_episodes.py first to transcribe episodes.")
        sys.exit(1)

    print(f"Loading transcripts from {TRANSCRIPTS_DIR}")
    podcast_parser = PodcastParser(podcast_name=PODCAST_NAME)
    episodes = podcast_parser.parse_directory(TRANSCRIPTS_DIR)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes found. Check that transcripts_topline/ contains .md files.")
        sys.exit(1)

    chunker = Chunker(chunk_size=500, chunk_overlap=75, min_chunk_size=50)
    records = build_records(episodes, chunker)
    print(f"Created {len(records)} chunks from {len(episodes)} episodes")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for r in records[:3]:
            print(f"\n  ID: {r['_id']}")
            print(f"  Title: {r['title']}")
            if r.get('guest'):
                print(f"  Guest: {r['guest']}")
            print(f"  Chunk: {r['chunk_index']}/{r['total_chunks']}")
            print(f"  Text: {r['text'][:100]}...")
        print(f"\nWould upsert {len(records)} records to {INDEX_NAME}/{NAMESPACE}")
        return

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

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
        if batch_num < total_batches:
            time.sleep(3)

    print(f"\nDone! Upserted {len(records)} records to {INDEX_NAME}/{NAMESPACE}")


if __name__ == "__main__":
    main()
