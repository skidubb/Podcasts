#!/usr/bin/env python3
"""Index podcast transcripts into a namespace of an integrated-inference
Pinecone index.

Generic version of index_topline_podcast.py: uses Pinecone integrated
inference (llama-text-embed-v2) — upserts text records directly, no OpenAI
embeddings needed. Works for any transcripts directory, index, and namespace.

Bulk backfill:
    python index_namespace_episodes.py \
        --transcripts-dir transcripts/light-reading \
        --index telecom-podcasts --namespace telecom \
        --podcast-name "Light Reading Podcasts" --id-prefix light-reading

Incremental (single episode, used by sync_episodes.py --index-new):
    python index_namespace_episodes.py ... --episode 42
"""

import argparse
import os
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

BATCH_SIZE = 20


def build_records(episodes, chunker, podcast_name: str, id_prefix: str):
    """Chunk episodes and build Pinecone records for integrated inference.

    IDs contain no spaces: GET-based SDK calls (fetch, list-by-prefix)
    silently return nothing for IDs with spaces.
    """
    records = []
    for episode in episodes:
        chunks = chunker.chunk_episode(episode)
        ep_part = f"{episode.episode_num:03d}" if episode.episode_num is not None else "000"
        for chunk in chunks:
            record = {
                "_id": f"{id_prefix}_{ep_part}_{chunk.chunk_index}",
                "text": chunk.text,
                "title": episode.title,
                "podcast_name": podcast_name,
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
    parser = argparse.ArgumentParser(
        description="Index podcast transcripts into a Pinecone namespace (integrated inference)"
    )
    parser.add_argument("--transcripts-dir", required=True, help="Directory of transcript .md files")
    parser.add_argument("--index", required=True, help="Pinecone index name (integrated inference)")
    parser.add_argument("--namespace", required=True, help="Pinecone namespace")
    parser.add_argument("--podcast-name", required=True, help="Podcast name stored in record metadata")
    parser.add_argument("--id-prefix", required=True, help="Record ID prefix (no spaces), e.g. light-reading")
    parser.add_argument("--episode", type=int, default=None, help="Index only this episode number")
    parser.add_argument("--dry-run", action="store_true", help="Preview without upserting")
    parser.add_argument("--start-batch", type=int, default=1, help="Resume from batch N (1-indexed)")
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    if not transcripts_dir.is_absolute():
        transcripts_dir = Path(__file__).parent / transcripts_dir
    if not transcripts_dir.exists():
        print(f"Transcripts directory not found: {transcripts_dir}")
        sys.exit(1)

    podcast_parser = PodcastParser(podcast_name=args.podcast_name)

    if args.episode is not None:
        matches = sorted(transcripts_dir.glob(f"{args.episode:03d}_*.md"))
        if not matches:
            print(f"No transcript found for episode {args.episode} in {transcripts_dir}")
            sys.exit(1)
        episodes = [podcast_parser.parse_file(matches[0])]
    else:
        print(f"Loading transcripts from {transcripts_dir}")
        episodes = podcast_parser.parse_directory(transcripts_dir)

    print(f"Loaded {len(episodes)} episode(s)")
    if not episodes:
        print(f"No episodes found. Check that {transcripts_dir} contains .md files.")
        sys.exit(1)

    chunker = Chunker(chunk_size=500, chunk_overlap=75, min_chunk_size=50)
    records = build_records(episodes, chunker, args.podcast_name, args.id_prefix)
    print(f"Created {len(records)} chunks from {len(episodes)} episode(s)")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for r in records[:3]:
            print(f"\n  ID: {r['_id']}")
            print(f"  Title: {r['title']}")
            if r.get('guest'):
                print(f"  Guest: {r['guest']}")
            print(f"  Chunk: {r['chunk_index']}/{r['total_chunks']}")
            print(f"  Text: {r['text'][:100]}...")
        print(f"\nWould upsert {len(records)} records to {args.index}/{args.namespace}")
        return

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index)

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
                index.upsert_records(namespace=args.namespace, records=batch)
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

    print(f"\nDone! Upserted {len(records)} records to {args.index}/{args.namespace}")


if __name__ == "__main__":
    main()
