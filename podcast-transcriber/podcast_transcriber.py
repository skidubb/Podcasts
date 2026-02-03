#!/usr/bin/env python3
"""
Podcast Transcriber

Downloads and transcribes podcast episodes from any RSS feed using OpenAI Whisper.
Outputs transcripts as markdown files.

Usage:
    pip install openai-whisper requests tqdm feedparser
    python podcast_transcriber.py <RSS_URL> [options]

Options:
    --episodes N, -e N    Limit to N episodes (default: all)
    --output PATH, -o     Output directory (default: ~/podcast_transcripts)
    --model MODEL, -m     Whisper model: tiny/base/small/medium/large (default: base)
    --recent, -r          Start with most recent episodes (default: oldest first)
"""

import os
import re
import sys
import time
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import whisper
except ImportError:
    print("Error: openai-whisper not installed. Run: pip install openai-whisper")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

try:
    import feedparser
except ImportError:
    print("Error: feedparser not installed. Run: pip install feedparser")
    sys.exit(1)


# Default Configuration
DEFAULT_OUTPUT_DIR = Path.home() / "podcast_transcripts"
DEFAULT_WHISPER_MODEL = "base"
RATE_LIMIT_SECONDS = 2
REQUEST_TIMEOUT = 60


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe podcast episodes from an RSS feed using Whisper"
    )
    parser.add_argument(
        "rss_url",
        help="RSS feed URL of the podcast"
    )
    parser.add_argument(
        "-e", "--episodes",
        type=int,
        default=None,
        help="Limit to N episodes (default: all)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model (default: {DEFAULT_WHISPER_MODEL})"
    )
    parser.add_argument(
        "-r", "--recent",
        action="store_true",
        help="Start with most recent episodes (default: oldest first)"
    )
    return parser.parse_args()


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


def fetch_episodes(rss_url: str, most_recent: bool = False):
    """Fetch episode list from RSS feed."""
    print(f"Fetching RSS feed from {rss_url}...")

    try:
        feed = feedparser.parse(rss_url)

        if feed.bozo and feed.bozo_exception:
            print(f"Warning: Feed parsing issue: {feed.bozo_exception}")

        # Get podcast title
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
            try:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt = datetime(*entry.published_parsed[:6])
                    formatted_date = dt.strftime("%b %d, %Y")
                else:
                    formatted_date = pub_date
            except Exception:
                formatted_date = pub_date

            duration = entry.get('itunes_duration', entry.get('duration', ''))

            episodes.append({
                'title': entry.get('title', 'Unknown Episode'),
                'mp3_url': mp3_url,
                'date': formatted_date,
                'duration': parse_duration(duration),
                'description': entry.get('summary', entry.get('description', '')),
            })

        # RSS feeds typically have newest first
        if not most_recent:
            episodes.reverse()

        print(f"Found {len(episodes)} episodes")
        return episodes, podcast_title

    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        return [], "Unknown Podcast"


def download_mp3(url: str, dest_path: Path) -> bool:
    """Download MP3 file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.RequestException as e:
        print(f"Download error: {e}")
        return False


def transcribe_audio(model, audio_path: Path) -> Optional[str]:
    """Transcribe audio file using Whisper."""
    try:
        print("Transcribing...")
        result = model.transcribe(str(audio_path), verbose=False)
        return result.get('text', '')
    except Exception as e:
        print(f"Transcription error: {e}")
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


def main():
    """Main transcription workflow."""
    args = parse_args()

    print("=" * 60)
    print("Podcast Transcriber")
    print("=" * 60)
    print()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")

    # Fetch episodes from RSS
    episodes, podcast_title = fetch_episodes(args.rss_url, args.recent)
    if not episodes:
        print("No episodes found. Exiting.")
        return

    # Limit episodes if configured
    if args.episodes:
        episodes = episodes[:args.episodes]
        order = "most recent" if args.recent else "oldest"
        print(f"Limited to {args.episodes} {order} episodes")

    # Load Whisper model
    print(f"\nLoading Whisper model '{args.model}'...")
    print("(This may take a moment on first run as the model downloads)")
    model = whisper.load_model(args.model)
    print("Model loaded successfully!")

    # Process statistics
    processed = 0
    skipped = 0
    failed = 0

    print(f"\nProcessing {len(episodes)} episodes...\n")

    for i, episode in enumerate(episodes, 1):
        output_filename = get_output_filename(episode, i)
        output_path = args.output / output_filename

        print("-" * 60)
        print(f"Episode {i}/{len(episodes)}: {episode['title']}")

        # Skip if already transcribed
        if output_path.exists():
            print(f"  -> Already transcribed, skipping")
            skipped += 1
            continue

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
            print(f"  -> Transcribing with Whisper ({args.model} model)...")
            transcript = transcribe_audio(model, tmp_path)

            if not transcript:
                print(f"  -> Failed to transcribe")
                failed += 1
                continue

            # Save markdown
            markdown = create_markdown(episode, transcript, i, podcast_title)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            print(f"  -> Saved: {output_filename}")
            processed += 1

        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

        # Rate limiting between episodes
        if i < len(episodes):
            time.sleep(RATE_LIMIT_SECONDS)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Podcast:            {podcast_title}")
    print(f"Total episodes:     {len(episodes)}")
    print(f"Newly transcribed:  {processed}")
    print(f"Already existed:    {skipped}")
    print(f"Failed:             {failed}")
    print(f"\nTranscripts saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
