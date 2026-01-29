"""LLM response generation with citations."""

from typing import Optional

from openai import OpenAI

from .config import Config
from .retriever import RetrievalResult


SYSTEM_PROMPT = """You are an expert analyst for the GTM AI Podcast, a show about go-to-market strategies, sales, marketing, and AI in B2B SaaS.

Your role is to answer questions by synthesizing insights from podcast transcripts. You have access to conversations with industry experts including founders, VPs of Sales, CMOs, RevOps leaders, and AI practitioners.

Guidelines:
1. Base your answers ONLY on the provided transcript excerpts
2. Cite sources using [Episode X - Guest Name] format
3. If information isn't in the excerpts, say so clearly
4. Synthesize insights across multiple excerpts when relevant
5. Be specific and actionable - these are practitioners seeking tactical advice
6. When guests disagree, present multiple perspectives with attribution

Response format:
- Start with a direct answer to the question
- Support with specific quotes or paraphrases from the transcripts
- End with key takeaways if the answer is complex"""


class Generator:
    """Generate RAG responses using LLM."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
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
        """Generate a response using retrieved context."""
        system_prompt = system_prompt or SYSTEM_PROMPT

        # Build context from results
        context = self._build_context(results, include_metadata)

        # Build user message
        user_message = f"""Question: {query}

Relevant transcript excerpts:

{context}

Based on these excerpts, please answer the question."""

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=1500,
        )

        return response.choices[0].message.content

    def generate_streaming(
        self,
        query: str,
        results: list[RetrievalResult],
        system_prompt: Optional[str] = None,
        include_metadata: bool = True,
    ):
        """Generate a streaming response."""
        system_prompt = system_prompt or SYSTEM_PROMPT
        context = self._build_context(results, include_metadata)

        user_message = f"""Question: {query}

Relevant transcript excerpts:

{context}

Based on these excerpts, please answer the question."""

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=1500,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
