"""Podcast transcript parser."""

import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Episode:
    """Parsed podcast episode."""
    episode_num: Optional[int]
    title: str
    guest: Optional[str]
    date: Optional[datetime]
    duration_minutes: Optional[int]
    file_path: Path
    transcript: str
    word_count: int
    podcast_name: str = "Podcast"

    def to_dict(self) -> dict:
        return {
            "episode_num": self.episode_num,
            "title": self.title,
            "guest": self.guest,
            "date": self.date.isoformat() if self.date else None,
            "duration_minutes": self.duration_minutes,
            "file_path": str(self.file_path),
            "word_count": self.word_count,
            "podcast_name": self.podcast_name,
        }


class PodcastParser:
    """Parser for podcast transcript markdown files."""

    # Common guest name patterns in podcast titles
    GUEST_PATTERNS = [
        r'with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # "with John Smith"
        r'featuring\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+on\s+',  # "John Smith on Topic"
        r'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(?:CEO|CRO|VP|CMO|Founder)',  # Name + title
    ]

    def __init__(self, podcast_name: str = "Podcast"):
        self.podcast_name = podcast_name

    def parse_file(self, file_path: Path) -> Episode:
        """Parse a markdown transcript file."""
        content = file_path.read_text(encoding='utf-8')

        episode_num = self._extract_episode_num(file_path)
        title = self._extract_title(content, file_path)
        guest = self._extract_guest(content, title)
        date = self._extract_date(content, episode_num)
        duration = self._extract_duration(content)
        transcript = self._clean_transcript(content)

        return Episode(
            episode_num=episode_num,
            title=title,
            guest=guest,
            date=date,
            duration_minutes=duration,
            file_path=file_path,
            transcript=transcript,
            word_count=len(transcript.split()),
            podcast_name=self.podcast_name,
        )

    def parse_directory(self, directory: Path) -> list[Episode]:
        """Parse all markdown files in a directory."""
        episodes = []
        md_files = sorted(directory.glob("*.md"))

        for file_path in md_files:
            try:
                episode = self.parse_file(file_path)
                episodes.append(episode)
            except Exception as e:
                print(f"Error parsing {file_path.name}: {e}")

        # Sort by episode number
        episodes.sort(key=lambda x: x.episode_num or 0)
        return episodes

    def _extract_episode_num(self, file_path: Path) -> Optional[int]:
        """Extract episode number from filename."""
        match = re.match(r'(\d+)', file_path.stem)
        return int(match.group(1)) if match else None

    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from H1 header or filename."""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        # Fallback to cleaned filename
        return file_path.stem.replace('_', ' ').title()

    def _extract_guest(self, content: str, title: str) -> Optional[str]:
        """Extract guest name from content or title."""
        # Try to find guest in metadata section
        guest_match = re.search(r'\*?\*?Guest:\*?\*?\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if guest_match:
            return guest_match.group(1).strip()

        # Try patterns in title
        for pattern in self.GUEST_PATTERNS:
            match = re.search(pattern, title)
            if match:
                return match.group(1).strip()

        # Try patterns in first 500 chars of content
        header_content = content[:500]
        for pattern in self.GUEST_PATTERNS:
            match = re.search(pattern, header_content)
            if match:
                return match.group(1).strip()

        return None

    def _extract_date(self, content: str, episode_num: Optional[int]) -> Optional[datetime]:
        """Extract date from content with multiple pattern fallbacks."""
        # Pattern 1: "Date: Jan 16, 2026" or "**Date:** Nov 13, 2025"
        abbrev_match = re.search(
            r'\*?\*?Date:\*?\*?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})',
            content, re.IGNORECASE
        )
        if abbrev_match:
            try:
                return datetime.strptime(
                    f"{abbrev_match.group(1)} {abbrev_match.group(2)} {abbrev_match.group(3)}",
                    '%b %d %Y'
                )
            except ValueError:
                pass

        # Pattern 2: Full month names "January 16, 2026"
        full_match = re.search(
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
            content, re.IGNORECASE
        )
        if full_match:
            try:
                return datetime.strptime(
                    f"{full_match.group(1)} {full_match.group(2)} {full_match.group(3)}",
                    '%B %d %Y'
                )
            except ValueError:
                pass

        # Pattern 3: ISO format "2024-01-16"
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
        if iso_match:
            try:
                return datetime.strptime(iso_match.group(1), '%Y-%m-%d')
            except ValueError:
                pass

        # Fallback: estimate from episode number (starting Sep 2023, weekly)
        if episode_num:
            from datetime import timedelta
            base_date = datetime(2023, 9, 1)
            return base_date + timedelta(days=(episode_num - 1) * 7)

        return None

    def _extract_duration(self, content: str) -> Optional[int]:
        """Extract duration in minutes from content."""
        # Pattern: "Duration: 45 min" or "45 minutes"
        duration_match = re.search(r'(?:Duration|Length):\s*(\d+)\s*(?:min|minutes)', content, re.IGNORECASE)
        if duration_match:
            return int(duration_match.group(1))

        # Pattern: "1:23:45" or "45:30" timestamps
        time_match = re.search(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b', content[:500])
        if time_match:
            hours = int(time_match.group(1)) if time_match.group(3) else 0
            minutes = int(time_match.group(2)) if time_match.group(3) else int(time_match.group(1))
            return hours * 60 + minutes if hours > 0 else minutes

        return None

    def _clean_transcript(self, content: str) -> str:
        """Clean transcript text for embedding."""
        # Remove YAML frontmatter
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

        # Remove markdown headers
        content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)

        # Remove speaker labels (e.g., "Host:", "Guest:")
        content = re.sub(r'^[A-Za-z]+:\s*', '', content, flags=re.MULTILINE)

        # Remove image markdown
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

        # Convert links to text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)

        # Remove bold/italic markers
        content = re.sub(r'\*+([^*]+)\*+', r'\1', content)

        # Normalize whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()
