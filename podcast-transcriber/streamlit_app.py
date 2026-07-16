"""Streamlit web app for podcast semantic search.

A shareable web interface for querying podcast transcripts using
Pinecone vector search and Claude for answer generation.

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.config import Config
from rag.embedder import Embedder
from rag.pinecone_indexer import PineconeIndexer
from rag.chunker import Chunk
from rag.retriever import RetrievalResult
from rag.generator import Generator


@st.cache_resource
def get_config():
    """Initialize configuration (cached)."""
    return Config()


@st.cache_resource
def get_embedder(_config):
    """Initialize embedder (cached)."""
    return Embedder(_config)


@st.cache_resource
def get_pinecone_indexer(_config):
    """Initialize Pinecone indexer (cached)."""
    return PineconeIndexer(_config)


@st.cache_resource
def get_generator(_config):
    """Initialize generator (cached)."""
    return Generator(_config)


def search_pinecone(
    query: str,
    embedder: Embedder,
    indexer: PineconeIndexer,
    top_k: int = 10,
    guest_filter: str = None,
    episode_filter: int = None,
    entity_filter: str = None,
) -> list[RetrievalResult]:
    """Search Pinecone and return results as RetrievalResult objects."""
    # Generate query embedding
    query_embedding = embedder.embed_query(query)

    # Build filter
    filter_dict = None
    if guest_filter or episode_filter or entity_filter:
        filter_dict = {}
        if guest_filter:
            filter_dict["guest"] = {"$eq": guest_filter}
        if episode_filter:
            filter_dict["episode_num"] = {"$eq": episode_filter}
        if entity_filter:
            # Equality on a list field matches any element; topics are stored lowercase
            filter_dict["$or"] = [
                {"people": {"$eq": entity_filter}},
                {"companies": {"$eq": entity_filter}},
                {"products": {"$eq": entity_filter}},
                {"topics": {"$eq": entity_filter.lower()}},
            ]

    # Search Pinecone
    matches = indexer.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter_dict=filter_dict,
    )

    # Convert to RetrievalResult
    results = []
    for i, match in enumerate(matches):
        chunk = indexer.chunk_from_match(match)
        result = RetrievalResult(
            chunk=chunk,
            score=match.score,
            rank=i + 1,
        )
        results.append(result)

    return results


def get_unique_guests(indexer: PineconeIndexer) -> list[str]:
    """Get list of unique guests from the index metadata."""
    # Note: For a production app, you'd want to store this separately
    # For now, we'll hardcode known guests or skip this feature
    return []


def main():
    # Get config for dynamic values
    config = get_config()
    podcast_name = config.podcast_name

    # Page configuration
    st.set_page_config(
        page_title=f"{podcast_name} Search",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title(f"🎙️ {podcast_name} Search")
    st.markdown(
        f"Search episodes of {podcast_name} using semantic search. "
        "Ask questions and get AI-powered answers with citations."
    )

    # Initialize components
    try:
        embedder = get_embedder(config)
        indexer = get_pinecone_indexer(config)
        generator = get_generator(config)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        st.info("Make sure your API keys are configured in Streamlit secrets or .env file.")
        return

    # Sidebar filters
    with st.sidebar:
        st.header("Search Settings")

        top_k = st.slider(
            "Number of sources",
            min_value=3,
            max_value=20,
            value=10,
            help="How many transcript excerpts to retrieve",
        )

        st.header("Filters")

        guest_filter = st.text_input(
            "Filter by guest name",
            placeholder="e.g., John Smith",
            help="Enter a guest name to filter results",
        )

        episode_filter = st.number_input(
            "Filter by episode number",
            min_value=0,
            max_value=200,
            value=0,
            help="Enter 0 for no filter",
        )

        entity_filter = st.text_input(
            "Filter by entity",
            placeholder="e.g., HubSpot, pricing, Jane Doe",
            help="Match a person, company, product/tool, or topic extracted from episodes",
        )

        st.divider()

        # Model selection
        model_choice = st.selectbox(
            "Answer model",
            options=["Claude Sonnet", "Claude Opus"],
            index=0,
            help="Sonnet is faster and cheaper; Opus is more thorough",
        )

        st.divider()

        st.markdown("**About**")
        st.markdown(
            "This app uses semantic search to find relevant excerpts "
            "from podcast transcripts, then uses Claude to synthesize an answer."
        )

        # Show index stats
        try:
            stats = indexer.get_stats()
            st.metric("Vectors indexed", f"{stats['total_vectors']:,}")
        except Exception:
            pass

    # Main search interface
    query = st.text_input(
        "Ask a question about the podcast",
        placeholder="What insights do experts share about this topic?",
    )

    # Example queries (generic)
    with st.expander("Example questions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - What are the key themes discussed?
            - What advice do guests give for beginners?
            - What trends do experts see emerging?
            """)
        with col2:
            st.markdown("""
            - What mistakes do guests say to avoid?
            - What tools or resources are recommended?
            - What predictions have guests made?
            """)

    if query:
        # Prepare filters
        guest = guest_filter if guest_filter else None
        episode = episode_filter if episode_filter > 0 else None
        entity = entity_filter.strip() if entity_filter and entity_filter.strip() else None

        with st.spinner("Searching transcripts..."):
            try:
                results = search_pinecone(
                    query=query,
                    embedder=embedder,
                    indexer=indexer,
                    top_k=top_k,
                    guest_filter=guest,
                    episode_filter=episode,
                    entity_filter=entity,
                )
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                return

        if not results:
            st.warning("No relevant excerpts found. Try a different query or remove filters.")
            return

        # Display sources in expander first
        with st.expander(f"📚 Sources ({len(results)} excerpts)", expanded=False):
            for result in results:
                chunk = result.chunk
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{chunk.citation()}**")
                        if chunk.date:
                            st.caption(f"📅 {chunk.date[:10]}")
                        chips = []
                        if chunk.topics:
                            chips.append("🏷️ " + " · ".join(chunk.topics[:6]))
                        mentions = (chunk.companies or [])[:4] + (chunk.products or [])[:4]
                        if mentions:
                            chips.append("💼 " + " · ".join(mentions))
                        if chips:
                            st.caption("  |  ".join(chips))
                    with col2:
                        st.metric("Relevance", f"{result.score:.2f}")

                    st.markdown(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)
                    st.divider()

        # Generate answer
        st.subheader("Answer")

        # Adjust model based on selection
        original_model = config.llm_model
        if model_choice == "Claude Sonnet":
            config.llm_model = "claude-sonnet-4-20250514"
        else:
            config.llm_model = "claude-opus-4-5-20251101"

        answer_placeholder = st.empty()
        full_response = ""

        try:
            # Stream the response
            for chunk in generator.generate_streaming(query, results):
                full_response += chunk
                answer_placeholder.markdown(full_response + "▌")

            answer_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Failed to generate answer: {str(e)}")
            # Show sources anyway
            st.info("Here are the relevant excerpts:")
            for result in results[:5]:
                st.markdown(f"**{result.chunk.citation()}**")
                st.markdown(result.chunk.text)
                st.divider()

        finally:
            # Restore original model
            config.llm_model = original_model

        # Citations summary
        st.subheader("Citations")
        citations = generator.get_citations(results)
        for cite in citations[:5]:
            ep_num = cite.get('episode_num', '')
            guest = cite.get('guest', 'Unknown Guest')
            title = cite.get('title', '')
            date = cite.get('date', '')[:10] if cite.get('date') else ''

            cite_text = f"**Episode {ep_num}**: {title}"
            if guest:
                cite_text += f" (Guest: {guest})"
            if date:
                cite_text += f" - {date}"
            st.markdown(cite_text)


if __name__ == "__main__":
    main()
