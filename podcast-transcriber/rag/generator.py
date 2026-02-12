"""LLM response generation with citations using Anthropic Claude."""

import time
from typing import Optional

import anthropic

from .config import Config
from .retriever import RetrievalResult


MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_system_prompt(config: Config) -> str:
    """Generate dynamic system prompt based on podcast configuration."""
    return f"""You are an expert analyst for {config.podcast_name}, {config.podcast_description}.

Your role is to answer questions by synthesizing insights from podcast transcripts. You have access to conversations with industry experts and thought leaders.

Guidelines:
1. Base your answers ONLY on the provided transcript excerpts
2. Cite sources using [Episode X - Guest Name] format
3. If information isn't in the excerpts, say so clearly
4. Synthesize insights across multiple excerpts when relevant
5. Be specific and actionable - these are practitioners seeking practical advice
6. When guests disagree, present multiple perspectives with attribution

Response format:
- Start with a direct answer to the question
- Support with specific quotes or paraphrases from the transcripts
- End with key takeaways if the answer is complex"""


class Generator:
    """Generate RAG responses using Claude."""

    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = config.llm_model
        self.max_context_tokens = config.max_context_tokens
        self.temperature = config.temperature

    def generate(
        self,
        query: str,
        results: list[RetrievalResult],
        system_prompt: Optional[str] = None,
        include_metadata: bool = True,
    ) -> str:
        """Generate a response using retrieved context with retry logic."""
        system_prompt = system_prompt or get_system_prompt(self.config)

        # Build context from results
        context = self._build_context(results, include_metadata)

        # Build user message
        user_message = f"""Question: {query}

Relevant transcript excerpts:

{context}

Based on these excerpts, please answer the question."""

        # Generate response with retries
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code == 529 or "overloaded" in str(e).lower():
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                raise

        if last_error:
            raise last_error

    def generate_streaming(
        self,
        query: str,
        results: list[RetrievalResult],
        system_prompt: Optional[str] = None,
        include_metadata: bool = True,
    ):
        """Generate a streaming response with retry logic."""
        system_prompt = system_prompt or get_system_prompt(self.config)
        context = self._build_context(results, include_metadata)

        user_message = f"""Question: {query}

Relevant transcript excerpts:

{context}

Based on these excerpts, please answer the question."""

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                return  # Success, exit retry loop
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code == 529 or "overloaded" in str(e).lower():
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                        continue
                raise  # Re-raise if not overloaded or out of retries

        if last_error:
            raise last_error

    def _build_context(
        self,
        results: list[RetrievalResult],
        include_metadata: bool = True,
    ) -> str:
        """Build context string from retrieval results."""
        context_parts = []

        for result in results:
            if include_metadata:
                header = f"[{result.chunk.citation()}]"
                if result.chunk.date:
                    header += f" ({result.chunk.date[:10]})"
                context_parts.append(f"{header}\n{result.chunk.text}")
            else:
                context_parts.append(result.chunk.text)

        return "\n\n---\n\n".join(context_parts)

    def get_citations(self, results: list[RetrievalResult]) -> list[dict]:
        """Extract citation information from results."""
        citations = []
        seen = set()

        for result in results:
            citation_key = (result.chunk.episode_num, result.chunk.guest)
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append({
                    'episode_num': result.chunk.episode_num,
                    'title': result.chunk.title,
                    'guest': result.chunk.guest,
                    'date': result.chunk.date,
                    'citation': result.chunk.citation(),
                })

        return citations
