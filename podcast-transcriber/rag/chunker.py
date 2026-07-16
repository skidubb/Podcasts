"""Document chunker with token-based sizing and overlap."""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Iterator

import tiktoken

from .parser import Episode


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    chunk_id: str
    text: str
    token_count: int

    # Episode metadata
    episode_num: Optional[int]
    title: str
    guest: Optional[str]
    date: Optional[str]  # ISO format
    podcast_name: str

    # Position metadata
    chunk_index: int
    total_chunks: int

    # Extracted entities (episode-level, copied onto every chunk)
    people: list = field(default_factory=list)
    companies: list = field(default_factory=list)
    products: list = field(default_factory=list)
    topics: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)

    def citation(self) -> str:
        """Generate citation string for this chunk."""
        parts = [self.podcast_name]
        if self.episode_num:
            parts.append(f"Ep. {self.episode_num}")
        if self.guest:
            parts.append(self.guest)
        return " - ".join(parts)


class Chunker:
    """Split documents into overlapping chunks with token counting."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        min_chunk_size: int = 50,
        encoding_name: str = "cl100k_base",  # GPT-4/embeddings encoding
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def chunk_episode(self, episode: Episode) -> list[Chunk]:
        """Split an episode transcript into chunks."""
        chunks = []
        text = episode.transcript

        # Split into sentences first
        sentences = self._split_sentences(text)

        current_chunk_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # First, save current chunk if any
                if current_chunk_sentences:
                    chunk_text = ' '.join(current_chunk_sentences)
                    if self.count_tokens(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk_sentences = []
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                for word in words:
                    word_token_count = self.count_tokens(word + ' ')
                    if word_tokens + word_token_count > self.chunk_size and word_chunk:
                        chunks.append(' '.join(word_chunk))
                        # Keep overlap
                        overlap_start = max(0, len(word_chunk) - self._estimate_words_for_tokens(self.chunk_overlap))
                        word_chunk = word_chunk[overlap_start:]
                        word_tokens = self.count_tokens(' '.join(word_chunk))
                    word_chunk.append(word)
                    word_tokens += word_token_count
                if word_chunk:
                    current_chunk_sentences = [' '.join(word_chunk)]
                    current_tokens = self.count_tokens(current_chunk_sentences[0])
                continue

            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                if self.count_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = self.count_tokens(' '.join(current_chunk_sentences))
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        # Convert to Chunk objects with metadata
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                chunk_id=f"{episode.podcast_name}_{episode.episode_num or 0}_{i}",
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                episode_num=episode.episode_num,
                title=episode.title,
                guest=episode.guest,
                date=episode.date.isoformat() if episode.date else None,
                podcast_name=episode.podcast_name,
                chunk_index=i,
                total_chunks=len(chunks),
                people=episode.people,
                companies=episode.companies,
                products=episode.products,
                topics=episode.topics,
            )
            result.append(chunk)

        return result

    def chunk_episodes(self, episodes: list[Episode]) -> list[Chunk]:
        """Chunk multiple episodes."""
        all_chunks = []
        for episode in episodes:
            chunks = self.chunk_episode(episode)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        """Get sentences for overlap from end of chunk."""
        if not sentences:
            return []

        overlap_sentences = []
        overlap_tokens = 0

        # Work backwards from end
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens

        return overlap_sentences

    def _estimate_words_for_tokens(self, tokens: int) -> int:
        """Rough estimate of words for a token count."""
        # Average ~1.3 tokens per word for English
        return int(tokens / 1.3)

    def save_chunks(self, chunks: list[Chunk], path: Path) -> None:
        """Save chunks to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')

    def load_chunks(self, path: Path) -> list[Chunk]:
        """Load chunks from JSONL file."""
        chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunks.append(Chunk.from_dict(data))
        return chunks
