"""RAG (Retrieval-Augmented Generation) module for podcast transcripts."""

import importlib

# Lazy imports (PEP 562) so light consumers (e.g. sync_episodes.py in CI)
# don't drag in heavy optional deps like faiss-cpu or anthropic.
_LAZY = {
    "Config": ".config",
    "PodcastParser": ".parser",
    "Chunker": ".chunker",
    "Embedder": ".embedder",
    "FAISSIndexer": ".indexer",
    "Retriever": ".retriever",
    "Generator": ".generator",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
