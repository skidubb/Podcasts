#!/usr/bin/env python3
"""Backfill entity metadata onto already-indexed episodes.

For each transcript in the configured transcripts directory, extracts
entities (cached in data/entities/<slug>/) and applies them to the
episode's existing Pinecone chunks via partial metadata updates —
vectors are never re-embedded.

Usage:
    python backfill_entities.py --dry-run       # extract + preview, no Pinecone writes
    python backfill_entities.py                 # full backfill for the configured podcast
    python backfill_entities.py --limit 5       # first 5 episodes only
    python backfill_entities.py --episode 270   # single episode

Per-podcast config comes from the environment (PODCAST_NAME,
PINECONE_INDEX_NAME, TRANSCRIPTS_DIR), same as sync_episodes.py.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.config import Config
from rag.parser import PodcastParser
from rag.entity_extractor import EntityExtractor, apply_entities
from rag.pinecone_indexer import PineconeIndexer


def main():
    parser = argparse.ArgumentParser(
        description="Backfill entity metadata onto indexed episodes (no re-embedding)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract entities but skip Pinecone updates")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N episodes")
    parser.add_argument("--episode", type=int, default=None,
                        help="Only process this episode number")
    args = parser.parse_args()

    config = Config()
    # No RSS/OpenAI needed here — only extraction (Anthropic) and Pinecone
    errors = []
    if not config.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY not set in environment")
    if not args.dry_run and not config.pinecone_api_key:
        errors.append("PINECONE_API_KEY not set in environment")
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(f"Podcast:     {config.podcast_name}")
    print(f"Index:       {config.pinecone_index_name}")
    print(f"Transcripts: {config.transcripts_dir}")
    print()

    podcast_parser = PodcastParser(podcast_name=config.podcast_name)
    episodes = podcast_parser.parse_directory(Path(config.transcripts_dir))
    if args.episode is not None:
        episodes = [ep for ep in episodes if ep.episode_num == args.episode]
    if args.limit:
        episodes = episodes[:args.limit]
    if not episodes:
        print("No matching transcripts found.")
        sys.exit(1)
    print(f"Episodes to process: {len(episodes)}")
    print()

    extractor = EntityExtractor(config)
    indexer = None if args.dry_run else PineconeIndexer(config)

    total_updated = 0
    failed = []
    for i, episode in enumerate(episodes, 1):
        entities = extractor.extract(episode)
        apply_entities(episode, entities)
        summary = (f"guest={entities.get('guest') or '-'} "
                   f"people={len(entities['people'])} companies={len(entities['companies'])} "
                   f"products={len(entities['products'])} topics={len(entities['topics'])}")
        if entities.get("error"):
            failed.append(episode.episode_num)
            print(f"[{i}/{len(episodes)}] Ep {episode.episode_num}: EXTRACTION FAILED — skipping update")
            continue

        if args.dry_run:
            print(f"[{i}/{len(episodes)}] Ep {episode.episode_num}: {summary} (dry run)")
            continue

        updated = indexer.update_episode_entities(
            podcast_name=episode.podcast_name,
            episode_num=episode.episode_num,
            entities=entities,
        )
        total_updated += updated
        marker = "" if updated else "  <-- no vectors found for this episode!"
        print(f"[{i}/{len(episodes)}] Ep {episode.episode_num}: {summary} -> {updated} vectors updated{marker}")

    print()
    print(f"Done. {total_updated} vectors updated across {len(episodes)} episodes.")
    if failed:
        print(f"Extraction failed for episodes: {failed} (re-run to retry; failures are not cached)")


if __name__ == "__main__":
    main()
