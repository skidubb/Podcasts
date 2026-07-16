"""LLM-based entity extraction for podcast episodes.

Extracts people, companies, products/tools, and topics from a transcript
in a single Claude call per episode, with a per-episode JSON cache so
rebuilds and re-runs never re-pay for extraction.
"""

import json
import re
from pathlib import Path
from typing import Optional

import anthropic

from .config import Config
from .parser import Episode
from .retry import retry_with_backoff, is_retryable_status

# Cheap model — extraction is a simple structured-output task
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

ENTITY_LIST_KEYS = ("people", "companies", "products", "topics")
MAX_LIST_ITEMS = 25
MAX_TRANSCRIPT_CHARS = 60_000  # ~15k tokens

EMPTY_ENTITIES = {
    "guest": None,
    "people": [],
    "companies": [],
    "products": [],
    "topics": [],
}

_PROMPT_TEMPLATE = """You are extracting entities from a podcast transcript for a search index.

Podcast: {podcast_name}
Episode title: {title}

Transcript (may be truncated):
<transcript>
{transcript}
</transcript>

Return ONLY a JSON object, no other text, with exactly these keys:
- "guest": the interviewed guest's full name as a string, or null if there is no guest (solo/news episode). Not the host.
- "people": names of people who appear or are substantively mentioned (include host and guest).
- "companies": companies, brands, agencies, shops, or practices mentioned substantively.
- "products": software, AI tools, platforms, or equipment discussed.
- "topics": 5-15 short subject-matter tags (lowercase, 1-3 words each) capturing what the episode is about, e.g. "pricing", "hiring", "patient acquisition".

Rules: proper names in their canonical form (e.g. "HubSpot" not "hubspot"), no duplicates, no honorifics like "Dr." in people entries, omit passing one-word mentions you are unsure about. Each list at most {max_items} items."""


class EntityExtractor:
    """Extract and cache per-episode entities via the Anthropic API."""

    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        self.config = config
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = EXTRACTION_MODEL
        # Cache lives in the repo (entities/ is tracked; data/ is gitignored)
        # so CI commits it alongside transcripts
        if cache_dir is None:
            slug = Path(config.transcripts_dir).name
            cache_dir = Path(__file__).parent.parent / "entities" / slug
        self.cache_dir = Path(cache_dir)

    def cache_path(self, episode_num: int) -> Path:
        return self.cache_dir / f"{episode_num}.json"

    def load_cached(self, episode_num: Optional[int]) -> Optional[dict]:
        """Return cached entities, or None if absent/errored (so it retries)."""
        if episode_num is None:
            return None
        path = self.cache_path(episode_num)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("error"):
            return None
        return _normalize(data)

    def extract(self, episode: Episode, use_cache: bool = True) -> dict:
        """Extract entities for an episode, using the cache when possible.

        Never raises on extraction failure: caches and returns empty entities
        with an "error" marker so indexing is never blocked. Error-marked
        cache entries are ignored by load_cached, so they retry next run.
        """
        if use_cache:
            cached = self.load_cached(episode.episode_num)
            if cached is not None:
                return cached

        prompt = _PROMPT_TEMPLATE.format(
            podcast_name=episode.podcast_name,
            title=episode.title,
            transcript=episode.transcript[:MAX_TRANSCRIPT_CHARS],
            max_items=MAX_LIST_ITEMS,
        )

        try:
            response = retry_with_backoff(
                self.client.messages.create,
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
                should_retry=is_retryable_status,
                label="Entity extraction",
            )
            entities = _parse_json_block(response.content[0].text)
        except Exception as e:
            print(f"  Entity extraction failed for episode {episode.episode_num}: {e}")
            entities = dict(EMPTY_ENTITIES, error=str(e))

        entities = _normalize(entities)
        self._save_cache(episode, entities)
        return entities

    def _save_cache(self, episode: Episode, entities: dict) -> None:
        if episode.episode_num is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(entities)
            payload["episode_num"] = episode.episode_num
            payload["title"] = episode.title
            payload["model"] = self.model
            self.cache_path(episode.episode_num).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            print(f"  Warning: could not write entity cache: {e}")


def apply_entities(episode: Episode, entities: dict) -> None:
    """Copy extracted entities onto an Episode (guest only if parser found none)."""
    episode.people = entities.get("people", [])
    episode.companies = entities.get("companies", [])
    episode.products = entities.get("products", [])
    episode.topics = entities.get("topics", [])
    if not episode.guest and entities.get("guest"):
        episode.guest = entities["guest"]


def _parse_json_block(text: str) -> dict:
    """Parse the JSON object out of an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    return json.loads(match.group(0))


def _normalize(data: dict) -> dict:
    """Coerce raw extraction output to a clean, capped, deduped schema."""
    result = {}
    guest = data.get("guest")
    result["guest"] = guest.strip() if isinstance(guest, str) and guest.strip() else None
    for key in ENTITY_LIST_KEYS:
        items = data.get(key) or []
        if not isinstance(items, list):
            items = []
        lowercase = key == "topics"
        result[key] = _dedupe(items, lowercase=lowercase)
    if data.get("error"):
        result["error"] = data["error"]
    return result


def _dedupe(items: list, lowercase: bool = False) -> list[str]:
    seen = {}
    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if lowercase:
            value = value.lower()
        if not value:
            continue
        key = value.lower()
        if key not in seen:
            seen[key] = value
    return list(seen.values())[:MAX_LIST_ITEMS]
