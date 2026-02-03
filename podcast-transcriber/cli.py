#!/usr/bin/env python3
"""Interactive CLI for querying the podcast knowledge base."""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint

from rag.config import Config
from rag.embedder import Embedder
from rag.indexer import FAISSIndexer
from rag.retriever import Retriever
from rag.generator import Generator

app = typer.Typer(
    name="podcast-rag",
    help="Query your podcast knowledge base using semantic search",
    add_completion=False,
)
console = Console()


def load_system() -> tuple[Config, Retriever, Generator]:
    """Load the RAG system components."""
    config = Config()

    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Load index
    indexer = FAISSIndexer(config)
    if not indexer.load():
        console.print("[red]Index not found. Run 'python build_index.py' first.[/red]")
        raise typer.Exit(1)

    embedder = Embedder(config)
    retriever = Retriever(config, embedder, indexer)
    generator = Generator(config)

    return config, retriever, generator


@app.command()
def chat(
    streaming: bool = typer.Option(True, "--streaming/--no-streaming", help="Stream responses"),
):
    """Interactive chat mode for querying the podcast knowledge base."""
    config, retriever, generator = load_system()

    console.print(Panel.fit(
        f"[bold blue]{config.podcast_name} Knowledge Base[/bold blue]\n"
        "Query your podcast using semantic search\n\n"
        "Commands: [dim]quit, exit, /filters, /sources[/dim]",
        border_style="blue",
    ))
    stats = retriever.indexer.get_stats()
    console.print(f"[dim]Loaded {stats['unique_episodes']} episodes, {stats['total_chunks']} chunks[/dim]\n")

    while True:
        try:
            query = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        query = query.strip()
        if not query:
            continue

        if query.lower() in ('quit', 'exit', 'q'):
            console.print("[dim]Goodbye![/dim]")
            break

        if query.lower() == '/filters':
            filters = retriever.get_available_filters()
            console.print("\n[bold]Available Filters:[/bold]")
            console.print(f"  Podcasts: {', '.join(filters['podcasts'])}")
            console.print(f"  Episodes: {min(filters['episodes'])}-{max(filters['episodes'])}")
            console.print(f"  Guests: {len(filters['guests'])} unique")
            console.print(f"  Date range: {filters['date_range']['min']} to {filters['date_range']['max']}")
            continue

        if query.lower() == '/sources':
            console.print("\n[dim]Last query sources will be shown after each response[/dim]")
            continue

        # Search and generate
        console.print()
        with console.status("[bold blue]Searching...[/bold blue]"):
            results = retriever.search(query, top_k=config.top_k)

        if not results:
            console.print("[yellow]No relevant results found.[/yellow]")
            continue

        console.print("[bold blue]Assistant[/bold blue]")

        if streaming:
            for token in generator.generate_streaming(query, results):
                console.print(token, end="")
            console.print()
        else:
            response = generator.generate(query, results)
            console.print(Markdown(response))

        # Show citations
        citations = generator.get_citations(results)
        console.print("\n[dim]Sources:[/dim]")
        for cite in citations[:5]:
            console.print(f"  [dim]• {cite['citation']}[/dim]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of chunks to retrieve"),
    guest: Optional[str] = typer.Option(None, "--guest", "-g", help="Filter by guest name"),
    episode: Optional[int] = typer.Option(None, "--episode", "-e", help="Filter by episode number"),
    raw: bool = typer.Option(False, "--raw", help="Show raw chunks without LLM generation"),
):
    """Single query mode."""
    config, retriever, generator = load_system()

    with console.status("[bold blue]Searching...[/bold blue]"):
        results = retriever.search(
            question,
            top_k=top_k,
            guest=guest,
            episode_num=episode,
        )

    if not results:
        console.print("[yellow]No relevant results found.[/yellow]")
        raise typer.Exit(0)

    if raw:
        # Show raw chunks
        for result in results:
            console.print(Panel(
                result.chunk.text,
                title=f"[bold]{result.chunk.citation()}[/bold] (score: {result.score:.3f})",
                border_style="dim",
            ))
    else:
        # Generate response
        with console.status("[bold blue]Generating response...[/bold blue]"):
            response = generator.generate(question, results)

        console.print(Markdown(response))

        # Show citations
        citations = generator.get_citations(results)
        console.print("\n[bold]Sources:[/bold]")
        for cite in citations:
            console.print(f"  • {cite['citation']}")


@app.command()
def stats():
    """Show index statistics."""
    config = Config()
    indexer = FAISSIndexer(config)

    if not indexer.load():
        console.print("[red]Index not found. Run 'python build_index.py' first.[/red]")
        raise typer.Exit(1)

    stats = indexer.get_stats()

    table = Table(title="Index Statistics", show_header=False, border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Status", "✓ Loaded" if stats['status'] == 'loaded' else "✗ Not loaded")
    table.add_row("Episodes", str(stats['unique_episodes']))
    table.add_row("Chunks", f"{stats['total_chunks']:,}")
    table.add_row("Vectors", f"{stats['total_vectors']:,}")
    table.add_row("Tokens", f"{stats['total_tokens']:,}")
    table.add_row("Unique Guests", str(stats['unique_guests']))
    table.add_row("Date Range", stats['date_range'] or "N/A")
    table.add_row("Est. Content", f"{stats['estimated_hours']} hours")
    table.add_row("Index Size", f"{stats['index_size_mb']} MB")

    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
):
    """Search without LLM generation (retrieval only)."""
    config, retriever, _ = load_system()

    with console.status("[bold blue]Searching...[/bold blue]"):
        results = retriever.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    for result in results:
        console.print(Panel(
            result.chunk.text[:500] + "..." if len(result.chunk.text) > 500 else result.chunk.text,
            title=f"[bold]{result.chunk.citation()}[/bold]",
            subtitle=f"Score: {result.score:.3f} | Rank: {result.rank}",
            border_style="blue" if result.rank == 1 else "dim",
        ))


if __name__ == "__main__":
    app()
