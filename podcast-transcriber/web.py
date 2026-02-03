#!/usr/bin/env python3
"""Web UI for podcast knowledge base."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag.config import Config
from rag.embedder import Embedder
from rag.pinecone_indexer import PineconeIndexer
from rag.retriever import PineconeRetriever
from rag.generator import Generator

# Initialize app
app = FastAPI(title="Podcast Knowledge Base")

# Global components (loaded on startup)
config: Config = None
retriever: PineconeRetriever = None
generator: Generator = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    guest: Optional[str] = None
    episode: Optional[int] = None


@app.on_event("startup")
async def startup():
    """Load RAG components on startup."""
    global config, retriever, generator

    config = Config()
    indexer = PineconeIndexer(config)

    embedder = Embedder(config)
    retriever = PineconeRetriever(config, embedder, indexer)
    generator = Generator(config)

    stats = indexer.get_stats()
    print(f"Connected to Pinecone index with {stats['total_vectors']} vectors")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the chat interface."""
    return get_html_template(config.podcast_name, config.podcast_description)


@app.get("/api/stats")
async def stats():
    """Get index statistics."""
    pinecone_stats = retriever.indexer.get_stats()
    # Return Pinecone stats in a format compatible with the UI
    return {
        'status': 'loaded',
        'total_vectors': pinecone_stats.get('total_vectors', 0),
        'total_chunks': pinecone_stats.get('total_vectors', 0),
        'unique_episodes': 91,  # Known from CLAUDE.md
        'unique_guests': 32,    # Known from CLAUDE.md
        'estimated_hours': 81,  # Known from CLAUDE.md
    }


@app.get("/api/insights")
async def insights():
    """Get detailed insights for analytics dashboard."""
    # Insights require local chunk data, which is not available with Pinecone
    # Return minimal data structure to prevent UI errors
    return {
        "top_guests": [],
        "all_guests": [],
        "guest_roles": {},
        "companies": [],
        "monthly_distribution": [],
        "chunk_sizes": [],
        "episodes": [],
        "total_guests": 32,  # Known values
        "total_episodes": 91,
        "message": "Detailed insights are not available when using Pinecone. Local chunk data is required for analytics."
    }


@app.get("/api/insights_disabled")
async def insights_disabled():
    """Original insights endpoint - disabled for Pinecone."""
    import re
    chunks = []  # retriever.indexer.chunks - not available with Pinecone

    # Build unique episodes with full metadata
    episodes = {}
    for c in chunks:
        if c.episode_num and c.episode_num not in episodes:
            episodes[c.episode_num] = {
                "num": c.episode_num,
                "title": c.title,
                "guest": c.guest,
                "date": c.date,
            }

    # Extract roles/companies from titles
    role_keywords = {
        "CEO": ["ceo", "chief executive"],
        "Founder": ["founder", "co-founder", "cofounder"],
        "VP Sales": ["vp sales", "vp of sales", "vice president sales", "head of sales"],
        "VP Marketing": ["vp marketing", "vp of marketing", "head of marketing"],
        "CRO": ["cro", "chief revenue"],
        "CMO": ["cmo", "chief marketing"],
        "RevOps": ["revops", "revenue operations", "sales ops"],
        "SDR/BDR": ["sdr", "bdr", "sales development"],
        "Consultant": ["consultant", "advisor", "coach"],
        "Analyst": ["analyst", "forrester", "gartner"],
    }

    guest_roles = defaultdict(list)
    companies = Counter()

    for ep in episodes.values():
        title_lower = (ep["title"] or "").lower()
        guest = ep["guest"] or ""

        # Extract company from "from Company" pattern
        company_match = re.search(r'from\s+([A-Z][A-Za-z0-9\s&]+?)(?:\s+on|\s+-|\s*$)', ep["title"] or "")
        if company_match:
            companies[company_match.group(1).strip()] += 1

        # Categorize by role
        found_role = False
        for role, keywords in role_keywords.items():
            if any(kw in title_lower for kw in keywords):
                guest_roles[role].append({"guest": guest, "episode": ep["num"], "title": ep["title"]})
                found_role = True
                break
        if not found_role and guest:
            guest_roles["Other"].append({"guest": guest, "episode": ep["num"], "title": ep["title"]})

    # Guest frequency by chunks
    guest_counts = Counter(c.guest for c in chunks if c.guest)
    top_guests = guest_counts.most_common(15)

    # All unique guests
    all_guests = sorted(set(c.guest for c in chunks if c.guest))

    # Episodes per month
    monthly = defaultdict(int)
    for c in chunks:
        if c.date:
            month = c.date[:7]
            monthly[month] += 1
    monthly_sorted = sorted(monthly.items())

    # Chunk size distribution
    size_buckets = {"<300": 0, "300-400": 0, "400-500": 0, "500-600": 0, ">600": 0}
    for c in chunks:
        t = c.token_count
        if t < 300:
            size_buckets["<300"] += 1
        elif t < 400:
            size_buckets["300-400"] += 1
        elif t < 500:
            size_buckets["400-500"] += 1
        elif t < 600:
            size_buckets["500-600"] += 1
        else:
            size_buckets[">600"] += 1

    return {
        "top_guests": [{"name": g, "chunks": c} for g, c in top_guests],
        "all_guests": all_guests,
        "guest_roles": {role: {"count": len(guests), "guests": guests} for role, guests in sorted(guest_roles.items(), key=lambda x: -len(x[1]))},
        "companies": [{"name": c, "count": n} for c, n in companies.most_common(15)],
        "monthly_distribution": [{"month": m, "chunks": c} for m, c in monthly_sorted],
        "chunk_sizes": [{"bucket": k, "count": v} for k, v in size_buckets.items()],
        "episodes": sorted(episodes.values(), key=lambda x: x["num"], reverse=True),
        "total_guests": len(all_guests),
        "total_episodes": len(episodes),
    }


@app.post("/api/query")
async def query(req: QueryRequest):
    """Query the knowledge base."""
    results = retriever.search(
        req.query,
        top_k=req.top_k,
        guest=req.guest,
        episode_num=req.episode,
    )

    if not results:
        return {"answer": "No relevant results found.", "sources": []}

    answer = generator.generate(req.query, results)
    citations = generator.get_citations(results)

    return {
        "answer": answer,
        "sources": citations,
    }


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """Stream query response."""
    results = retriever.search(
        req.query,
        top_k=req.top_k,
        guest=req.guest,
        episode_num=req.episode,
    )

    if not results:
        async def empty():
            yield "data: " + json.dumps({"type": "content", "content": "No relevant results found."}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "sources": []}) + "\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    citations = generator.get_citations(results)

    async def generate():
        for token in generator.generate_streaming(req.query, results):
            yield "data: " + json.dumps({"type": "content", "content": token}) + "\n\n"
        yield "data: " + json.dumps({"type": "done", "sources": citations}) + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def get_html_template(podcast_name: str, podcast_description: str) -> str:
    """Generate HTML template with dynamic podcast name."""
    return HTML_TEMPLATE.replace(
        "{{PODCAST_NAME}}", podcast_name
    ).replace(
        "{{PODCAST_DESCRIPTION}}", podcast_description
    )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{PODCAST_NAME}} Knowledge Base</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #ffffff;
            color: #1f2937;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: #f8fafc;
            padding: 1rem 2rem;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-left h1 {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
        }

        .header-left p {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }

        .tabs {
            display: flex;
            gap: 0.5rem;
        }

        .tab {
            background: transparent;
            border: 1px solid #d1d5db;
            color: #6b7280;
            padding: 0.5rem 1.25rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tab:hover {
            background: #f3f4f6;
            color: #1f2937;
        }

        .tab.active {
            background: #2563eb;
            border-color: #2563eb;
            color: #fff;
        }

        .container {
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .view {
            display: none;
            flex: 1;
            overflow: hidden;
            flex-direction: column;
        }

        .view.active {
            display: flex;
        }

        /* Chat View */
        #chat-view main {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }

        .message {
            margin-bottom: 1.5rem;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            text-align: right;
        }

        .message.user .bubble {
            background: #2563eb;
            color: white;
            display: inline-block;
            padding: 0.75rem 1rem;
            border-radius: 1rem 1rem 0.25rem 1rem;
            max-width: 80%;
        }

        .message.assistant .bubble {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 1rem 1.25rem;
            border-radius: 0.25rem 1rem 1rem 1rem;
            line-height: 1.7;
            white-space: pre-wrap;
        }

        .sources {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            font-size: 0.8rem;
            color: #6b7280;
        }

        .sources-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #4b5563;
        }

        .source {
            display: inline-block;
            background: #e5e7eb;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0.25rem 0.25rem 0;
            font-size: 0.75rem;
        }

        .typing span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #9ca3af;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .typing span:nth-child(1) { animation-delay: -0.32s; }
        .typing span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        #chat-view footer {
            background: #f8fafc;
            padding: 1rem 2rem;
            border-top: 1px solid #e2e8f0;
        }

        .input-container {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 0.75rem;
        }

        input[type="text"] {
            flex: 1;
            background: #ffffff;
            border: 1px solid #d1d5db;
            color: #1f2937;
            padding: 0.875rem 1rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }

        input[type="text"]:focus {
            border-color: #2563eb;
        }

        input[type="text"]::placeholder {
            color: #9ca3af;
        }

        button.send-btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 0.875rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }

        button.send-btn:hover {
            background: #1d4ed8;
        }

        button.send-btn:disabled {
            background: #d1d5db;
            cursor: not-allowed;
        }

        .welcome {
            text-align: center;
            padding: 3rem;
            color: #6b7280;
        }

        .welcome h2 {
            color: #374151;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .welcome p {
            margin-bottom: 1.5rem;
        }

        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
        }

        .example {
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .example:hover {
            background: #e5e7eb;
            border-color: #d1d5db;
        }

        /* Insights View */
        #insights-view {
            padding: 2rem;
            overflow-y: auto;
            background: #f8fafc;
        }

        .insights-grid {
            max-width: 1400px;
            margin: 0 auto;
        }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .stat-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 1.25rem;
        }

        .stat-card .label {
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }

        .stat-card .value {
            font-size: 2rem;
            font-weight: 600;
            color: #1f2937;
        }

        .stat-card .subtext {
            font-size: 0.75rem;
            color: #9ca3af;
            margin-top: 0.25rem;
        }

        .stat-card.accent {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-color: #bfdbfe;
        }

        .stat-card.accent .value {
            color: #2563eb;
        }

        .charts-row {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .chart-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 1.25rem;
        }

        .chart-card h3 {
            font-size: 0.9rem;
            font-weight: 600;
            color: #4b5563;
            margin-bottom: 1rem;
        }

        .episodes-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .episode-list {
            max-height: 300px;
            overflow-y: auto;
        }

        .episode-item {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            border-bottom: 1px solid #f3f4f6;
        }

        .episode-item:last-child {
            border-bottom: none;
        }

        .episode-num {
            background: #f3f4f6;
            color: #6b7280;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            margin-right: 0.75rem;
            min-width: 36px;
            text-align: center;
        }

        .episode-info {
            flex: 1;
            min-width: 0;
        }

        .episode-title {
            font-size: 0.85rem;
            color: #374151;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .episode-meta {
            font-size: 0.75rem;
            color: #9ca3af;
            margin-top: 0.125rem;
        }

        .guest-bar {
            display: flex;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .guest-name {
            width: 140px;
            font-size: 0.8rem;
            color: #6b7280;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .guest-bar-bg {
            flex: 1;
            height: 24px;
            background: #f3f4f6;
            border-radius: 0.25rem;
            overflow: hidden;
        }

        .guest-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #2563eb, #3b82f6);
            border-radius: 0.25rem;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 0.5rem;
            font-size: 0.7rem;
            color: #fff;
            font-weight: 500;
        }

        @media (max-width: 1200px) {
            .stats-row {
                grid-template-columns: repeat(2, 1fr);
            }
            .charts-row, .episodes-row {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 600px) {
            .stats-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-left">
            <h1>{{PODCAST_NAME}} Knowledge Base</h1>
            <p id="stats-line">Loading...</p>
        </div>
        <div class="tabs">
            <button class="tab active" data-view="chat-view">Chat</button>
            <button class="tab" data-view="insights-view">Insights</button>
        </div>
    </header>

    <div class="container">
        <!-- Chat View -->
        <div id="chat-view" class="view active">
            <main id="chat">
                <div class="welcome" id="welcome">
                    <h2>Ask anything about {{PODCAST_NAME}}</h2>
                    <p>{{PODCAST_DESCRIPTION}}</p>
                    <div class="examples" id="examples"></div>
                </div>
            </main>
            <footer>
                <div class="input-container">
                    <input type="text" id="input" placeholder="Ask a question..." autofocus>
                    <button class="send-btn" id="send">Send</button>
                </div>
            </footer>
        </div>

        <!-- Insights View -->
        <div id="insights-view" class="view">
            <div class="insights-grid">
                <div class="stats-row" id="stats-cards"></div>
                <div class="charts-row">
                    <div class="chart-card">
                        <h3>Content Volume Over Time</h3>
                        <div id="timeline-chart"></div>
                    </div>
                    <div class="chart-card">
                        <h3>Chunk Size Distribution</h3>
                        <div id="size-chart"></div>
                    </div>
                </div>
                <div class="episodes-row">
                    <div class="chart-card">
                        <h3>Top Guests by Coverage</h3>
                        <div id="guests-chart"></div>
                    </div>
                    <div class="chart-card">
                        <h3>Recent Episodes</h3>
                        <div class="episode-list" id="episodes-list"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.view).classList.add('active');

                if (tab.dataset.view === 'insights-view' && !window.insightsLoaded) {
                    loadInsights();
                }
            });
        });

        // Chat functionality
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const examples = document.getElementById('examples');
        let isLoading = false;

        const exampleQuestions = [
            "What are the key themes discussed?",
            "What advice do guests give for beginners?",
            "What trends do experts see emerging?",
            "What tools or resources are recommended?"
        ];

        exampleQuestions.forEach(q => {
            const btn = document.createElement('div');
            btn.className = 'example';
            btn.textContent = q;
            btn.addEventListener('click', () => {
                input.value = q;
                sendMessage();
            });
            examples.appendChild(btn);
        });

        // Load basic stats
        fetch('/api/stats')
            .then(r => r.json())
            .then(stats => {
                document.getElementById('stats-line').textContent =
                    `${stats.unique_episodes} episodes · ${stats.estimated_hours} hours · ${stats.total_chunks.toLocaleString()} chunks`;
            });

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !isLoading) sendMessage();
        });
        sendBtn.addEventListener('click', () => { if (!isLoading) sendMessage(); });

        function addUserMessage(content) {
            const welcome = document.getElementById('welcome');
            if (welcome) welcome.remove();
            const div = document.createElement('div');
            div.className = 'message user';
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.textContent = content;
            div.appendChild(bubble);
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function addAssistantMessage() {
            const div = document.createElement('div');
            div.className = 'message assistant';
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            div.appendChild(bubble);
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return bubble;
        }

        function addTypingIndicator() {
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.id = 'typing';
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            const typing = document.createElement('div');
            typing.className = 'typing';
            for (let i = 0; i < 3; i++) typing.appendChild(document.createElement('span'));
            bubble.appendChild(typing);
            div.appendChild(bubble);
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function removeTypingIndicator() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }

        function addSources(bubble, sources) {
            if (sources.length === 0) return;
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            const title = document.createElement('div');
            title.className = 'sources-title';
            title.textContent = 'Sources';
            sourcesDiv.appendChild(title);
            sources.forEach(s => {
                const span = document.createElement('span');
                span.className = 'source';
                span.textContent = s.citation;
                sourcesDiv.appendChild(span);
            });
            bubble.appendChild(sourcesDiv);
        }

        async function sendMessage() {
            const query = input.value.trim();
            if (!query || isLoading) return;
            isLoading = true;
            sendBtn.disabled = true;
            input.value = '';
            addUserMessage(query);
            addTypingIndicator();

            try {
                const response = await fetch('/api/query/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: 10 })
                });
                removeTypingIndicator();
                const bubble = addAssistantMessage();
                let fullContent = '';
                let sources = [];
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const text = decoder.decode(value);
                    const lines = text.split('\\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.type === 'content') {
                                    fullContent += data.content;
                                    bubble.textContent = fullContent;
                                    chat.scrollTop = chat.scrollHeight;
                                } else if (data.type === 'done') {
                                    sources = data.sources;
                                }
                            } catch (e) {}
                        }
                    }
                }
                addSources(bubble, sources);
            } catch (error) {
                removeTypingIndicator();
                const bubble = addAssistantMessage();
                bubble.textContent = 'Error: ' + error.message;
            }
            isLoading = false;
            sendBtn.disabled = false;
            input.focus();
        }

        // Insights functionality
        async function loadInsights() {
            window.insightsLoaded = true;
            const data = await fetch('/api/insights').then(r => r.json());
            const stats = await fetch('/api/stats').then(r => r.json());

            // Stats cards
            const statsCards = document.getElementById('stats-cards');
            const cards = [
                { label: 'Total Episodes', value: stats.unique_episodes, subtext: 'Indexed in knowledge base', accent: true },
                { label: 'Content Hours', value: stats.estimated_hours, subtext: 'Of expert conversation' },
                { label: 'Unique Guests', value: data.total_guests, subtext: 'Industry practitioners' },
                { label: 'Search Chunks', value: stats.total_chunks.toLocaleString(), subtext: 'Semantic segments' },
            ];
            cards.forEach(c => {
                const card = document.createElement('div');
                card.className = 'stat-card' + (c.accent ? ' accent' : '');
                const label = document.createElement('div');
                label.className = 'label';
                label.textContent = c.label;
                const value = document.createElement('div');
                value.className = 'value';
                value.textContent = c.value;
                const subtext = document.createElement('div');
                subtext.className = 'subtext';
                subtext.textContent = c.subtext;
                card.appendChild(label);
                card.appendChild(value);
                card.appendChild(subtext);
                statsCards.appendChild(card);
            });

            // Timeline chart
            new ApexCharts(document.getElementById('timeline-chart'), {
                chart: { type: 'area', height: 250, background: 'transparent', toolbar: { show: false } },
                series: [{ name: 'Chunks', data: data.monthly_distribution.map(m => m.chunks) }],
                xaxis: { categories: data.monthly_distribution.map(m => m.month), labels: { style: { colors: '#6b7280' } } },
                yaxis: { labels: { style: { colors: '#6b7280' } } },
                colors: ['#2563eb'],
                fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.4, opacityTo: 0.1 } },
                stroke: { curve: 'smooth', width: 2 },
                grid: { borderColor: '#e5e7eb' },
                theme: { mode: 'light' },
                tooltip: { theme: 'light' }
            }).render();

            // Size distribution chart
            new ApexCharts(document.getElementById('size-chart'), {
                chart: { type: 'donut', height: 250, background: 'transparent' },
                series: data.chunk_sizes.map(s => s.count),
                labels: data.chunk_sizes.map(s => s.bucket + ' tokens'),
                colors: ['#1e40af', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
                legend: { position: 'bottom', labels: { colors: '#6b7280' } },
                plotOptions: { pie: { donut: { size: '60%' } } },
                theme: { mode: 'light' }
            }).render();

            // Guest bars
            const guestsChart = document.getElementById('guests-chart');
            const maxChunks = Math.max(...data.top_guests.map(g => g.chunks));
            data.top_guests.forEach(g => {
                const bar = document.createElement('div');
                bar.className = 'guest-bar';
                const name = document.createElement('div');
                name.className = 'guest-name';
                name.textContent = g.name;
                const barBg = document.createElement('div');
                barBg.className = 'guest-bar-bg';
                const fill = document.createElement('div');
                fill.className = 'guest-bar-fill';
                fill.style.width = (g.chunks / maxChunks * 100) + '%';
                fill.textContent = g.chunks;
                barBg.appendChild(fill);
                bar.appendChild(name);
                bar.appendChild(barBg);
                guestsChart.appendChild(bar);
            });

            // Episodes list
            const episodesList = document.getElementById('episodes-list');
            data.episodes.forEach(ep => {
                const item = document.createElement('div');
                item.className = 'episode-item';
                const num = document.createElement('div');
                num.className = 'episode-num';
                num.textContent = '#' + ep.num;
                const info = document.createElement('div');
                info.className = 'episode-info';
                const title = document.createElement('div');
                title.className = 'episode-title';
                title.textContent = ep.title;
                const meta = document.createElement('div');
                meta.className = 'episode-meta';
                meta.textContent = (ep.guest || 'No guest') + (ep.date ? ' · ' + ep.date.slice(0, 10) : '');
                info.appendChild(title);
                info.appendChild(meta);
                item.appendChild(num);
                item.appendChild(info);
                episodesList.appendChild(item);
            });
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
