"""Entity vocabulary built from the committed per-episode entity caches.

Pinecone only offers exact matching ($eq/$in) on list metadata, so a filter
for "ChatGPT" silently misses chunks tagged "Chat GPT". The extractor runs
per episode with no shared vocabulary, so those surface variants are
unavoidable on the write path.

This module reads `entities/<slug>/*.json` — the same cache the indexer
wrote from, tracked in git — to answer two questions at query time:

    values(field)        what can a user actually filter on?
    expand(field, value) which stored surface forms mean this?

`expand` merges only variants that are identical once case and punctuation
are stripped ("Zoom Info" -> "ZoomInfo", "CARFAX" -> "Carfax"). It does not
do fuzzy matching: "Catherine Maley"/"Katherine Maley" are transcription
variants of one person, but "Michael Kane"/"Michael Caine" are two people,
and nothing at this layer can tell them apart. Collapsing those needs a
reviewed alias map, not a similarity threshold.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from .config import Config

ENTITY_FIELDS = ("people", "companies", "products", "topics")

# Only "topics" is lowercased at extraction time (see entity_extractor).
LOWERCASE_FIELDS = frozenset({"topics"})


def normalize(value: str) -> str:
    """Collapse case and punctuation so surface variants share a key."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


class EntityVocabulary:
    """Filterable entity values observed across a podcast's episodes."""

    def __init__(self, entities_dir: Optional[Path] = None):
        self.entities_dir = Path(entities_dir) if entities_dir else None
        # field -> surface form -> number of episodes mentioning it
        self._counts: dict[str, Counter] = {f: Counter() for f in ENTITY_FIELDS}
        # field -> normalized key -> [surface forms]
        self._variants: dict[str, dict[str, list[str]]] = {f: {} for f in ENTITY_FIELDS}
        self._load()

    @classmethod
    def for_config(cls, config: Config) -> "EntityVocabulary":
        """Locate the cache directory the way EntityExtractor does.

        The slug is the transcripts directory name, NOT the slugified
        podcast name — "The Face Podcast with Alex Pike" caches under
        `entities/the-face-podcast`. Deriving it any other way silently
        yields an empty vocabulary and no variant expansion.
        """
        slug = Path(config.transcripts_dir).name
        return cls(Path(__file__).parent.parent / "entities" / slug)

    def _load(self) -> None:
        if not self.entities_dir or not self.entities_dir.is_dir():
            return

        for path in sorted(self.entities_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for field in ENTITY_FIELDS:
                for item in data.get(field) or []:
                    if not isinstance(item, str):
                        continue
                    value = item.strip()
                    if not value:
                        continue
                    self._counts[field][value] += 1
                    self._variants[field].setdefault(normalize(value), [])
                    forms = self._variants[field][normalize(value)]
                    if value not in forms:
                        forms.append(value)

    @property
    def is_empty(self) -> bool:
        return not any(self._counts[f] for f in ENTITY_FIELDS)

    def values(self, field: str, limit: Optional[int] = None) -> list[str]:
        """Surface forms for a field, most-mentioned first.

        Variants of the same normalized entity collapse to the most common
        form, so the caller shows one option rather than five spellings.
        """
        if field not in self._counts:
            raise ValueError(f"Unknown entity field: {field}")

        best: dict[str, tuple[int, str]] = {}
        for value, count in self._counts[field].items():
            key = normalize(value)
            if key not in best or count > best[key][0]:
                best[key] = (count, value)

        ranked = sorted(best.values(), key=lambda pair: (-pair[0], pair[1].lower()))
        values = [value for _, value in ranked]
        return values[:limit] if limit else values

    def expand(self, field: str, value: str) -> list[str]:
        """All stored surface forms equivalent to `value` under normalization.

        Falls back to the caller's own value (plus a lowercased form for
        topics, which are stored lowercase) when nothing is known — an
        unlisted entity should still be filterable.
        """
        if field not in self._variants:
            raise ValueError(f"Unknown entity field: {field}")

        value = (value or "").strip()
        if not value:
            return []

        forms = list(self._variants[field].get(normalize(value), ()))
        if forms:
            return forms

        fallback = [value]
        if field in LOWERCASE_FIELDS and value.lower() != value:
            fallback.append(value.lower())
        return fallback

    def expand_all(self, field: str, values: Iterable[str]) -> list[str]:
        """Union of `expand` over several values, order-stable and deduped."""
        out: list[str] = []
        for value in values or ():
            for form in self.expand(field, value):
                if form not in out:
                    out.append(form)
        return out
