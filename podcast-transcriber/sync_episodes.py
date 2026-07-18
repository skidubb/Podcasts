#!/usr/bin/env python3
"""Sync podcast episodes from RSS feed.

This script:
1. Fetches RSS feed to get latest episode list
2. Compares against existing transcripts
3. Downloads and transcribes any missing episodes
4. Optionally rebuilds the Pinecone index

Usage:
    python sync_episodes.py                    # Check for and transcribe new episodes
    python sync_episodes.py --dry-run          # Show what would be done without doing it
    python sync_episodes.py --rebuild-index    # Also rebuild Pinecone index after transcribing
    python sync_episodes.py --force-episode 92 # Force re-transcribe specific episode
"""

import os
import re
import sys
import time
import argparse
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import feedparser
except ImportError:
    print("Error: feedparser not installed. Run: pip install feedparser")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Progress bars will not be shown.")
    tqdm = None

# Import config
from rag.config import Config
from rag.retry import retry_with_backoff, is_retryable_status

# Constants
RATE_LIMIT_SECONDS = 2
REQUEST_TIMEOUT = 60
SCRIPT_DIR = Path(__file__).parent


def get_config() -> Config:
    """Get configuration, with validation."""
    config = Config()

    if not config.rss_feed_url:
        print("Error: RSS_FEED_URL not set in .env file")
        print("Please set RSS_FEED_URL to your podcast's RSS feed URL")
        sys.exit(1)

    return config


def slugify(text: str, max_length: int = 50) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s]+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text[:max_length].rstrip('_')


def parse_duration(duration_str: Optional[str]) -> str:
    """Parse duration from various formats to human-readable string."""
    if not duration_str:
        return "Unknown"

    if ':' in str(duration_str):
        parts = str(duration_str).split(':')
        if len(parts) == 3:
            hours, mins, secs = map(int, parts)
            if hours > 0:
                return f"{hours}h {mins}m"
            return f"{mins} min"
        elif len(parts) == 2:
            mins, secs = map(int, parts)
            return f"{mins} min"

    try:
        total_secs = int(duration_str)
        mins = total_secs // 60
        return f"{mins} min"
    except (ValueError, TypeError):
        return str(duration_str)


def fetch_rss_episodes(rss_url: str):
    """Fetch episode list from RSS feed."""
    print(f"Fetching RSS feed from {rss_url}...")

    try:
        feed = feedparser.parse(rss_url)

        if feed.bozo and feed.bozo_exception:
            print(f"Warning: Feed parsing issue: {feed.bozo_exception}")

        podcast_title = feed.feed.get('title', 'Unknown Podcast')
        print(f"Podcast: {podcast_title}")

        episodes = []
        for entry in feed.entries:
            # Find the MP3 enclosure
            mp3_url = None
            for link in entry.get('links', []):
                if link.get('type', '').startswith('audio/') or link.get('href', '').endswith('.mp3'):
                    mp3_url = link.get('href')
                    break

            if not mp3_url:
                for enclosure in entry.get('enclosures', []):
                    if enclosure.get('type', '').startswith('audio/') or enclosure.get('url', '').endswith('.mp3'):
                        mp3_url = enclosure.get('url')
                        break

            if not mp3_url:
                print(f"Warning: No MP3 found for episode: {entry.get('title', 'Unknown')}")
                continue

            # Parse publication date
            pub_date = entry.get('published', entry.get('updated', ''))
            formatted_date = pub_date
            parsed_date = None
            try:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    parsed_date = datetime(*entry.published_parsed[:6])
                    formatted_date = parsed_date.strftime("%b %d, %Y")
            except Exception:
                pass

            duration = entry.get('itunes_duration', entry.get('duration', ''))

            episodes.append({
                'title': entry.get('title', 'Unknown Episode'),
                'guid': entry.get('id', '') or mp3_url,
                'mp3_url': mp3_url,
                'date': formatted_date,
                'parsed_date': parsed_date,
                'duration': parse_duration(duration),
                'description': entry.get('summary', entry.get('description', '')),
            })

        # RSS feeds typically have newest first, reverse to get chronological order
        episodes.reverse()

        print(f"Found {len(episodes)} episodes in feed")
        return episodes, podcast_title

    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        return [], "Unknown Podcast"


def get_existing_episodes(transcripts_dir: Path) -> tuple[dict, set]:
    """Get existing transcripts: {episode_num: path} and the set of known GUIDs.

    GUIDs are read from the '**GUID:** ...' line in each transcript's header.
    Older transcripts predate GUID tracking and only contribute a number.
    """
    existing = {}
    guids = set()

    if not transcripts_dir.exists():
        return existing, guids

    for md_file in transcripts_dir.glob("*.md"):
        match = re.match(r'(\d+)_', md_file.name)
        if not match:
            continue
        episode_num = int(match.group(1))
        existing[episode_num] = md_file
        try:
            header = md_file.read_text(encoding='utf-8')[:2000]
            guid_match = re.search(r'^\*\*GUID:\*\*\s*(\S+)', header, re.MULTILINE)
            if guid_match:
                guids.add(guid_match.group(1))
        except OSError:
            pass

    return existing, guids


def find_missing_episodes(rss_episodes: list, existing: dict, known_guids: set) -> list:
    """Find episodes in RSS that don't have local transcripts.

    An episode counts as existing if its GUID is known, or (for transcripts
    that predate GUID tracking) if its feed position matches an existing
    episode number. New episodes are numbered sequentially past the current
    max so numbers never shift when the feed window changes.
    """
    missing = []
    next_num = max(existing.keys(), default=0) + 1

    for i, episode in enumerate(rss_episodes, 1):
        if episode['guid'] in known_guids:
            continue
        if i in existing:
            continue  # legacy transcript without a stored GUID
        missing.append((next_num, episode))
        next_num += 1

    return missing


# Some CDNs (e.g. Acast) return 403 to the default python-requests
# user agent; a browser UA is accepted.
DOWNLOAD_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
}


def _download_mp3_once(url: str, dest_path: Path) -> None:
    """Download MP3 file with optional progress bar (single attempt)."""
    response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT,
                            headers=DOWNLOAD_HEADERS)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        if tqdm and total_size > 0:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def download_mp3(url: str, dest_path: Path) -> bool:
    """Download MP3 with retry and exponential backoff."""
    try:
        retry_with_backoff(
            _download_mp3_once, url, dest_path,
            retryable=(requests.RequestException,),
            should_retry=is_retryable_status,
            label="download",
        )
        return True
    except requests.RequestException as e:
        print(f"Download error: {e}")
        return False


def transcribe_audio(audio_path: Path, model: str = "base") -> Optional[str]:
    """Transcribe audio file using Whisper."""
    try:
        import whisper
        print(f"Transcribing with Whisper ({model} model)...")
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(str(audio_path), verbose=False)
        return result.get('text', '')
    except ImportError:
        print("Error: openai-whisper not installed. Run: pip install openai-whisper")
        return None
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def transcribe_audio_cloud(audio_path: Path, openai_client) -> Optional[str]:
    """Transcribe audio via the OpenAI Whisper API, compressing if needed."""
    from transcribe_cloud import compress_if_needed, transcribe_episode

    try:
        audio_path = compress_if_needed(audio_path)
        transcript = retry_with_backoff(
            transcribe_episode, openai_client, audio_path,
            should_retry=is_retryable_status,
            label="cloud transcription",
        )
        return transcript
    except Exception as e:
        print(f"Cloud transcription error: {e}")
        return None


def create_markdown(episode: dict, transcript: str, episode_num: int, podcast_title: str) -> str:
    """Generate markdown content for an episode."""
    description = re.sub(r'<[^>]+>', '', episode.get('description', ''))
    description = description.strip()[:500]

    content = f"""# {episode['title']}

**Podcast:** {podcast_title}
**Episode:** {episode_num}
**Date:** {episode['date']}
**Duration:** {episode['duration']}
**GUID:** {episode.get('guid', '')}
**MP3:** [{episode['title']}]({episode['mp3_url']})

---

## Description

{description}

---

## Transcript

{transcript}
"""
    return content


def get_output_filename(episode: dict, episode_num: int) -> str:
    """Generate output filename for an episode."""
    slug = slugify(episode['title'])
    return f"{episode_num:03d}_{slug}.md"


def index_new_episode(episode_num: int, config=None) -> bool:
    """Incrementally index a single episode into Pinecone (no full rebuild)."""
    if config is not None and config.pinecone_namespace:
        # Integrated-inference index + namespace: no OpenAI embeddings and
        # no entity-extraction layer on this path.
        script = SCRIPT_DIR / "index_namespace_episodes.py"
        cmd = [
            sys.executable, str(script),
            "--transcripts-dir", str(config.transcripts_dir),
            "--index", config.pinecone_index_name,
            "--namespace", config.pinecone_namespace,
            "--podcast-name", config.podcast_name,
            "--id-prefix", config.transcripts_dir.name,
            "--episode", str(episode_num),
        ]
    else:
        script = SCRIPT_DIR / "add_episode.py"
        cmd = [sys.executable, str(script), str(episode_num)]

    if not script.exists():
        print(f"Error: {script.name} not found at {script}")
        return False

    try:
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error indexing episode {episode_num}: {e}")
        return False


def rebuild_pinecone_index():
    """Rebuild the Pinecone index."""
    print("\n" + "=" * 60)
    print("Rebuilding Pinecone index...")
    print("=" * 60)

    build_script = SCRIPT_DIR / "build_pinecone_index.py"
    if not build_script.exists():
        print(f"Error: build_pinecone_index.py not found at {build_script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(build_script), "--rebuild"],
            cwd=str(SCRIPT_DIR),
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        return False


def sync_episodes(
    dry_run: bool = False,
    rebuild_index: bool = False,
    index_new: bool = False,
    force_episode: Optional[int] = None,
    whisper_model: str = None,
    limit: Optional[int] = None,
    cloud: bool = False,
    since: Optional[datetime] = None,
    skip_title: Optional[str] = None,
):
    """Main sync function."""
    print("=" * 60)
    print("Podcast Episode Sync")
    print("=" * 60)
    print()

    # Load configuration
    config = get_config()

    # --rebuild-index drives build_pinecone_index.py (OpenAI embeddings,
    # whole-index rebuild) which is incompatible with namespace-mode
    # integrated-inference indexes.
    if rebuild_index and config.pinecone_namespace:
        print("Warning: --rebuild-index is not supported when PINECONE_NAMESPACE "
              "is set; use index_namespace_episodes.py instead. Ignoring.")
        rebuild_index = False

    # Get whisper model from config or argument
    if whisper_model is None:
        whisper_model = config.whisper_model

    # Ensure transcripts directory exists
    transcripts_dir = config.transcripts_dir
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Transcripts directory: {transcripts_dir}")

    # Fetch RSS feed
    rss_episodes, podcast_title = fetch_rss_episodes(config.rss_feed_url)
    if not rss_episodes:
        print("No episodes found in RSS feed. Exiting.")
        return

    # Date-window filter. Episodes with no parseable date are excluded too:
    # fail closed so a feed date-format change can't trigger a mass backfill.
    if since:
        undated = sum(1 for e in rss_episodes if not e['parsed_date'])
        kept = [e for e in rss_episodes if e['parsed_date'] and e['parsed_date'] >= since]
        if undated:
            print(f"Warning: {undated} episodes have no parseable date; excluded by --since")
        print(f"--since {since.date()}: keeping {len(kept)} of {len(rss_episodes)} episodes")
        rss_episodes = kept

    # Title-pattern filter (e.g. skip a sub-series within the feed)
    if skip_title:
        pattern = re.compile(skip_title, re.IGNORECASE)
        kept = [e for e in rss_episodes if not pattern.search(e['title'])]
        print(f"--skip-title {skip_title!r}: skipping {len(rss_episodes) - len(kept)} episodes")
        rss_episodes = kept

    if not rss_episodes:
        print("No episodes remain after filtering. Exiting.")
        return

    # Get existing transcripts
    existing, known_guids = get_existing_episodes(transcripts_dir)
    print(f"Existing transcripts: {len(existing)} ({len(known_guids)} with GUIDs)")

    # Find missing episodes
    if force_episode:
        # Force specific episode
        if force_episode <= len(rss_episodes):
            missing = [(force_episode, rss_episodes[force_episode - 1])]
            print(f"Forcing re-transcription of episode {force_episode}")
        else:
            print(f"Episode {force_episode} not found in RSS feed (max: {len(rss_episodes)})")
            return
    else:
        missing = find_missing_episodes(rss_episodes, existing, known_guids)

    # Apply limit (take most recent N)
    if limit and len(missing) > limit:
        missing = missing[-limit:]
        print(f"  (limited to most recent {limit})")

    if not missing:
        print("\nAll episodes are already transcribed!")
        if rebuild_index:
            rebuild_pinecone_index()
        return

    print(f"\nMissing episodes: {len(missing)}")
    for episode_num, episode in missing:
        print(f"  - Episode {episode_num}: {episode['title'][:50]}...")

    if dry_run:
        print("\nDry run - no changes made.")
        return

    # Set up transcription backend
    model = None
    openai_client = None
    if cloud:
        print("\nUsing OpenAI Whisper API for transcription (cloud mode)")
        if not config.openai_api_key:
            print("Error: OPENAI_API_KEY not set (required for --cloud)")
            return
        from openai import OpenAI
        openai_client = OpenAI(api_key=config.openai_api_key)
    else:
        # Load Whisper model once
        print(f"\nLoading Whisper model '{whisper_model}'...")
        try:
            import whisper
            model = whisper.load_model(whisper_model)
            print("Model loaded successfully!")
        except ImportError:
            print("Error: openai-whisper not installed. Run: pip install openai-whisper")
            return
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return

    # Process missing episodes
    processed = 0
    failed = 0
    indexed = 0

    print(f"\nProcessing {len(missing)} episodes...\n")

    for idx, (episode_num, episode) in enumerate(missing):
        output_filename = get_output_filename(episode, episode_num)
        output_path = transcripts_dir / output_filename

        print("-" * 60)
        print(f"Episode {episode_num}: {episode['title']}")

        # Download MP3 to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            print(f"  -> Downloading from: {episode['mp3_url'][:60]}...")
            if not download_mp3(episode['mp3_url'], tmp_path):
                print(f"  -> Failed to download")
                failed += 1
                continue

            # Transcribe
            if cloud:
                print(f"  -> Transcribing via OpenAI API...")
                transcript = transcribe_audio_cloud(tmp_path, openai_client)
            else:
                print(f"  -> Transcribing with Whisper ({whisper_model} model)...")
                result = model.transcribe(str(tmp_path), verbose=False)
                transcript = result.get('text', '')

            if not transcript:
                print(f"  -> Failed to transcribe")
                failed += 1
                continue

            # Save markdown
            markdown = create_markdown(episode, transcript, episode_num, podcast_title)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            print(f"  -> Saved: {output_filename}")
            processed += 1

            # Incrementally index the new episode (no full rebuild)
            if index_new:
                print(f"  -> Indexing episode {episode_num} into Pinecone...")
                if index_new_episode(episode_num, config):
                    indexed += 1
                else:
                    print(f"  -> Warning: indexing failed for episode {episode_num}")

        finally:
            # Clean up temp files (compression may leave a .compressed.mp3)
            for leftover in (tmp_path, tmp_path.with_suffix('.compressed.mp3')):
                if leftover.exists():
                    leftover.unlink()

        # Rate limiting between episodes
        if idx < len(missing) - 1:
            time.sleep(RATE_LIMIT_SECONDS)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Podcast:            {podcast_title}")
    print(f"RSS episodes:       {len(rss_episodes)}")
    print(f"Missing episodes:   {len(missing)}")
    print(f"Newly transcribed:  {processed}")
    print(f"Failed:             {failed}")
    if index_new:
        print(f"Indexed:            {indexed}")
    print(f"\nTranscripts saved to: {transcripts_dir}")

    # Rebuild index if requested and we processed episodes
    if rebuild_index and processed > 0:
        rebuild_pinecone_index()
    elif rebuild_index and processed == 0:
        print("\nNo new episodes to index.")

    print("=" * 60)


def main():
    """Parse arguments and run sync."""
    parser = argparse.ArgumentParser(
        description="Sync podcast episodes from RSS feed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Fully rebuild the Pinecone index after transcribing (manual use)"
    )
    parser.add_argument(
        "--index-new",
        action="store_true",
        help="Incrementally index each newly transcribed episode into Pinecone"
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Transcribe via the OpenAI Whisper API instead of local Whisper"
    )
    parser.add_argument(
        "--force-episode",
        type=int,
        help="Force re-transcribe a specific episode number"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the N most recent missing episodes"
    )
    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model (default: from config or 'base')"
    )
    parser.add_argument(
        "--since",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=None,
        metavar="YYYY-MM-DD",
        help="Only consider episodes published on or after this date"
    )
    parser.add_argument(
        "--skip-title",
        default=None,
        metavar="REGEX",
        help="Skip episodes whose title matches this regex (case-insensitive)"
    )

    args = parser.parse_args()

    sync_episodes(
        dry_run=args.dry_run,
        rebuild_index=args.rebuild_index,
        index_new=args.index_new,
        force_episode=args.force_episode,
        whisper_model=args.model,
        limit=args.limit,
        cloud=args.cloud,
        since=args.since,
        skip_title=args.skip_title,
    )


if __name__ == "__main__":
    main()
