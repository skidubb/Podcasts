#!/usr/bin/env python3
"""Query a podcast Pinecone index from the command line.

Used by the .claude/agents aesthetic-practice specialists (rev consultant, etc.)
to ground answers in podcast transcripts. Standalone: needs only OPENAI_API_KEY
and PINECONE_API_KEY (from env, repo .env files, or ~/.zshrc exports).

Examples:
    python query_kb.py "how should a medspa price Botox memberships"
    python query_kb.py --index beauty-and-the-biz --guest "Catherine Maley" "patient financing"
    python query_kb.py --topic "consult conversion" --top-k 15 --json "increase consult close rate"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_INDEX = "beauty-and-the-biz"

KEY_SOURCES = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.home() / ".zshrc",
]


def load_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name].strip()
    pattern = re.compile(rf'^(?:export\s+)?{name}\s*=\s*["\']?([^"\'\n]+)["\']?\s*$')
    for path in KEY_SOURCES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            m = pattern.match(line.strip())
            if m:
                return m.group(1).strip()
    sys.exit(f"ERROR: {name} not found in environment, .env files, or ~/.zshrc")


def main() -> None:
    p = argparse.ArgumentParser(description="Semantic search over a podcast Pinecone index")
    p.add_argument("query", help="Natural-language search query")
    p.add_argument("--index", default=DEFAULT_INDEX, help=f"Pinecone index name (default: {DEFAULT_INDEX})")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--guest", help="Exact guest name filter")
    p.add_argument("--episode", type=int, help="Episode number filter")
    p.add_argument("--person", help="Filter: person mentioned in chunk")
    p.add_argument("--company", help="Filter: company mentioned in chunk")
    p.add_argument("--product", help="Filter: product mentioned in chunk")
    p.add_argument("--topic", help="Filter: topic tag on chunk")
    p.add_argument("--json", action="store_true", help="Emit raw JSON instead of markdown")
    args = p.parse_args()

    from openai import OpenAI
    from pinecone import Pinecone

    embedding = (
        OpenAI(api_key=load_key("OPENAI_API_KEY"))
        .embeddings.create(model=EMBED_MODEL, input=[args.query])
        .data[0]
        .embedding
    )

    filter_dict = {}
    if args.guest:
        filter_dict["guest"] = {"$eq": args.guest}
    if args.episode is not None:
        filter_dict["episode_num"] = {"$eq": args.episode}
    for field, value in (
        ("people", args.person),
        ("companies", args.company),
        ("products", args.product),
        ("topics", args.topic),
    ):
        if value:
            filter_dict[field] = {"$in": [value]}

    index = Pinecone(api_key=load_key("PINECONE_API_KEY")).Index(args.index)
    result = index.query(
        vector=embedding,
        top_k=args.top_k,
        filter=filter_dict or None,
        include_metadata=True,
    )

    matches = result.matches
    if args.json:
        print(json.dumps(
            [{"id": m.id, "score": round(m.score, 4), "metadata": dict(m.metadata or {})} for m in matches],
            indent=2,
        ))
        return

    if not matches:
        print("No results.")
        return

    for m in matches:
        md = m.metadata or {}
        header = f"Ep {int(md.get('episode_num', 0))}: {md.get('title', '?')}"
        if md.get("guest"):
            header += f" — guest: {md['guest']}"
        if md.get("date"):
            header += f" ({md['date']})"
        print(f"### {header}  [score {m.score:.3f}]")
        for field in ("topics", "companies", "products"):
            if md.get(field):
                print(f"_{field}: {', '.join(md[field])}_")
        print()
        print(md.get("text", "").strip())
        print("\n---\n")


if __name__ == "__main__":
    main()
