# Podcast Transcriber

Tools for downloading, transcribing, and analyzing podcast episodes.

## Features

- **podcast_transcriber.py** - Downloads and transcribes podcast episodes from any RSS feed using OpenAI Whisper
- **theme_analysis.py** - Analyzes transcripts for themes, topics, and trends using predefined taxonomies

## Installation

```bash
cd "Podcast Transcriber"
python -m venv podcast_venv
source podcast_venv/bin/activate  # On Windows: podcast_venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Transcribe a Podcast

```bash
python podcast_transcriber.py <RSS_URL> [options]
```

**Options:**
- `-e N, --episodes N` - Limit to N episodes (default: all)
- `-o PATH, --output PATH` - Output directory (default: ~/podcast_transcripts)
- `-m MODEL, --model MODEL` - Whisper model: tiny/base/small/medium/large (default: base)
- `-r, --recent` - Start with most recent episodes (default: oldest first)

**Example:**
```bash
python podcast_transcriber.py "https://example.com/feed.xml" -e 10 -m base
```

### Analyze Themes

```bash
python theme_analysis.py
```

Generates charts and reports analyzing topic distribution and trends across episodes.

## Output

Transcripts are saved as markdown files with metadata including:
- Episode title
- Publication date
- Duration
- MP3 link
- Full transcript

## Requirements

- Python 3.8+
- FFmpeg (required by Whisper for audio processing)
