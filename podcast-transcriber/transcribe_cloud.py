#!/usr/bin/env python3
"""Transcribe podcast episodes using OpenAI's Whisper API (cloud).

Handles the full pipeline: fetch RSS, find missing episodes, download,
compress if needed, transcribe via API, save as markdown.

Usage:
    python transcribe_cloud.py --dry-run          # Preview missing episodes
    python transcribe_cloud.py                    # Transcribe all missing
    python transcribe_cloud.py --limit 50         # Most recent 50 missing
    python transcribe_cloud.py --episode 93       # Specific episode only
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import feedparser
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Config from env (stripped — trailing whitespace in secrets breaks HTTP headers)
RSS_URL = os.getenv("RSS_FEED_URL", "").strip() or None
PODCAST_NAME = os.getenv("PODCAST_NAME", "Podcast").strip()
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "./transcripts").strip())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None

OPENAI_MAX_FILE_SIZE = 24 * 1024 * 1024  # 24MB (API limit is 25MB)

def get_ffmpeg_path():
    """Get ffmpeg binary path, preferring imageio-ffmpeg bundled binary."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def slugify(text: str, max_length: int = 50) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s]+', '_', text)
    return text[:max_length].rstrip('_')


def fetch_rss(rss_url):
    """Fetch and parse RSS feed, return (episodes_list, podcast_title)."""
    print(f"Fetching RSS feed from {rss_url}...")
    feed = feedparser.parse(rss_url)
    podcast_title = feed.feed.get('title', PODCAST_NAME)
    print(f"Podcast: {podcast_title}")

    episodes = []
    for entry in reversed(feed.entries):  # chronological order
        mp3_url = None
        for link in entry.get('links', []):
            if link.get('type', '').startswith('audio/') or link.get('href', '').endswith('.mp3'):
                mp3_url = link.get('href')
                break
        if not mp3_url:
            for enc in entry.get('enclosures', []):
                if enc.get('type', '').startswith('audio/') or enc.get('url', '').endswith('.mp3'):
                    mp3_url = enc.get('url')
                    break
        if not mp3_url:
            continue

        pub_date = entry.get('published', '')
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                parsed = datetime(*entry.published_parsed[:6])
                pub_date = parsed.strftime("%b %d, %Y")
        except Exception:
            pass

        episodes.append({
            'title': entry.get('title', 'Unknown'),
            'mp3_url': mp3_url,
            'date': pub_date,
            'duration': entry.get('itunes_duration', 'Unknown'),
            'description': re.sub(r'<[^>]+>', '', entry.get('summary', entry.get('description', '')))[:500],
        })

    print(f"Found {len(episodes)} episodes in feed")
    return episodes, podcast_title


def get_existing(transcripts_dir):
    """Return set of episode numbers that already have transcripts."""
    existing = set()
    if transcripts_dir.exists():
        for f in transcripts_dir.glob("*.md"):
            match = re.match(r'(\d+)_', f.name)
            if match:
                existing.add(int(match.group(1)))
    return existing


def download_mp3(url, dest_path):
    """Download MP3 file."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    print(f"\r  Downloaded: {downloaded * 100 / total:.0f}%", end="", flush=True)
    print()


def compress_if_needed(mp3_path):
    """Compress MP3 if over OpenAI's size limit. Returns path to use."""
    file_size = mp3_path.stat().st_size
    if file_size <= OPENAI_MAX_FILE_SIZE:
        return mp3_path

    # Calculate target bitrate to fit under 24MB
    # Estimate duration from file size (assume ~128kbps source)
    est_duration_sec = file_size / (128 * 1024 / 8)
    target_bitrate = int((OPENAI_MAX_FILE_SIZE * 8) / est_duration_sec * 0.85)  # 85% safety margin
    target_bitrate = max(16000, min(target_bitrate, 48000))  # clamp 16k-48k

    print(f"  File is {file_size / 1024 / 1024:.1f}MB, compressing to ~{target_bitrate // 1000}kbps...")
    compressed = mp3_path.with_suffix('.compressed.mp3')
    result = subprocess.run([
        get_ffmpeg_path(), '-i', str(mp3_path),
        '-b:a', str(target_bitrate), '-ac', '1', '-ar', '16000', '-y',
        str(compressed)
    ], capture_output=True)
    if result.returncode != 0:
        print(f"  Warning: ffmpeg compression failed, using original")
        return mp3_path

    mp3_path.unlink()
    new_size = compressed.stat().st_size
    print(f"  Compressed to {new_size / 1024 / 1024:.1f}MB")
    return compressed


def transcribe_episode(client, mp3_path):
    """Transcribe audio via OpenAI Whisper API."""
    with open(mp3_path, 'rb') as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return transcript


def save_markdown(episode, transcript, episode_num, podcast_title, transcripts_dir):
    """Save transcript as markdown file."""
    markdown = f"""# {episode['title']}

**Podcast:** {podcast_title}
**Episode:** {episode_num}
**Date:** {episode['date']}
**Duration:** {episode['duration']}
**MP3:** [{episode['title']}]({episode['mp3_url']})

---

## Description

{episode['description']}

---

## Transcript

{transcript}
"""
    filename = f"{episode_num:03d}_{slugify(episode['title'])}.md"
    output_path = transcripts_dir / filename
    output_path.write_text(markdown, encoding='utf-8')
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe podcast episodes via OpenAI Whisper API")
    parser.add_argument("--dry-run", action="store_true", help="Preview without transcribing")
    parser.add_argument("--limit", type=int, help="Only process the N most recent missing episodes")
    parser.add_argument("--episode", type=int, help="Transcribe a specific episode number")
    args = parser.parse_args()

    if not RSS_URL:
        print("Error: RSS_FEED_URL not set in .env")
        sys.exit(1)

    episodes, podcast_title = fetch_rss(RSS_URL)
    if not episodes:
        print("No episodes found.")
        sys.exit(1)

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = get_existing(TRANSCRIPTS_DIR)
    print(f"Existing transcripts: {len(existing)}")

    # Determine which episodes to process
    if args.episode:
        if args.episode > len(episodes):
            print(f"Episode {args.episode} not found (max: {len(episodes)})")
            sys.exit(1)
        to_process = [(args.episode, episodes[args.episode - 1])]
    else:
        to_process = [(i, ep) for i, ep in enumerate(episodes, 1) if i not in existing]
        if args.limit and len(to_process) > args.limit:
            to_process = to_process[-args.limit:]
            print(f"  (limited to most recent {args.limit})")

    print(f"\nEpisodes to transcribe: {len(to_process)}")
    for num, ep in to_process:
        print(f"  - Episode {num}: {ep['title'][:60]}...")

    if args.dry_run:
        print("\nDry run - no changes made.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    processed = 0
    failed = 0

    for num, episode in to_process:
        print(f"\n{'=' * 60}")
        print(f"Episode {num}: {episode['title']}")

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Download
            print(f"  Downloading...")
            download_mp3(episode['mp3_url'], tmp_path)

            # Compress if needed
            tmp_path = compress_if_needed(tmp_path)

            # Transcribe
            print(f"  Transcribing via OpenAI API...")
            transcript = transcribe_episode(client, tmp_path)

            if not transcript:
                print(f"  Failed - empty transcript")
                failed += 1
                continue

            # Save
            output = save_markdown(episode, transcript, num, podcast_title, TRANSCRIPTS_DIR)
            print(f"  Saved: {output.name}")
            processed += 1

        except Exception as e:
            print(f"  Error: {e}")
            failed += 1
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Brief pause between API calls
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"Done! Transcribed: {processed}, Failed: {failed}")
    print(f"Transcripts in: {TRANSCRIPTS_DIR}")


if __name__ == "__main__":
    main()
