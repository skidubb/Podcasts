#!/usr/bin/env python3
"""
GTM AI Podcast Theme Extraction & Trend Analysis

Analyzes podcast transcripts using two predefined taxonomies:
1. Tools & Tactics - AI tools, technologies, and tactical approaches
2. Functions & Roles - Business functions and job roles discussed
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
TRANSCRIPTS_DIR = Path("/Users/scottewalt/Documents/Podcasts/podcast_transcripts/GTM_AI_Podcast")
OUTPUT_DIR = Path("/Users/scottewalt/Documents/Podcast Transcriber/output")
MIN_CHUNK_WORDS = 50
MAX_CHUNK_WORDS = 300

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TAXONOMY DEFINITIONS
# =============================================================================

# Tools & Tactics Taxonomy
TOOLS_TACTICS_TAXONOMY = {
    'AI Agents': {
        'keywords': ['agent', 'agents', 'autonomous', 'agentic', 'ai agent', 'ai agents'],
        'description': 'Autonomous AI agents and agentic workflows'
    },
    'LLMs & ChatGPT': {
        'keywords': ['chatgpt', 'gpt', 'llm', 'llms', 'claude', 'gemini', 'openai',
                     'anthropic', 'language model', 'large language'],
        'description': 'Large language models and chat interfaces'
    },
    'Prompting': {
        'keywords': ['prompt', 'prompts', 'prompting', 'prompt engineering', 'system prompt',
                     'few shot', 'chain of thought', 'zero shot'],
        'description': 'Prompt engineering and techniques'
    },
    'Automation & Workflows': {
        'keywords': ['automate', 'automation', 'workflow', 'workflows', 'zapier', 'make',
                     'n8n', 'automated', 'automating', 'orchestration'],
        'description': 'Workflow automation and orchestration tools'
    },
    'CRM & Sales Tools': {
        'keywords': ['salesforce', 'hubspot', 'crm', 'gong', 'outreach', 'salesloft',
                     'apollo', 'zoominfo', 'linkedin sales', 'sales navigator', 'clari'],
        'description': 'CRM platforms and sales tech stack'
    },
    'Email & Outreach': {
        'keywords': ['cold email', 'outreach', 'email sequence', 'cadence', 'cold call',
                     'prospecting', 'outbound', 'inbound', 'sequences'],
        'description': 'Email and outreach tactics'
    },
    'Content & Copy': {
        'keywords': ['content', 'copywriting', 'copy', 'writing', 'blog', 'video',
                     'podcast', 'webinar', 'ebook', 'whitepaper', 'case study'],
        'description': 'Content creation and copywriting'
    },
    'Data & Analytics': {
        'keywords': ['data', 'analytics', 'metrics', 'kpi', 'kpis', 'dashboard',
                     'reporting', 'insights', 'analysis', 'forecast', 'forecasting'],
        'description': 'Data analysis and reporting'
    },
    'Personalization': {
        'keywords': ['personalize', 'personalization', 'personalized', 'custom',
                     'customize', 'tailor', 'tailored', 'segment', 'segmentation'],
        'description': 'Personalization and segmentation tactics'
    },
    'Research & Intel': {
        'keywords': ['research', 'intelligence', 'intent', 'signals', 'enrichment',
                     'technographics', 'firmographics', 'buyer intent', 'signal'],
        'description': 'Research and sales intelligence'
    },
}

# Functions & Roles Taxonomy
FUNCTIONS_ROLES_TAXONOMY = {
    'Sales': {
        'keywords': ['sales', 'selling', 'seller', 'sellers', 'deal', 'deals',
                     'close', 'closing', 'quota', 'pipeline', 'opportunity'],
        'description': 'Sales function and activities'
    },
    'SDR/BDR': {
        'keywords': ['sdr', 'bdr', 'sdrs', 'bdrs', 'prospecting', 'outbound rep',
                     'sales development', 'business development', 'appointment setting'],
        'description': 'Sales/Business Development Representatives'
    },
    'Account Executive': {
        'keywords': ['account executive', 'ae', 'aes', 'closer', 'closers',
                     'full cycle', 'enterprise rep', 'strategic rep'],
        'description': 'Account Executives and closers'
    },
    'Marketing': {
        'keywords': ['marketing', 'marketer', 'marketers', 'demand gen', 'brand',
                     'campaign', 'campaigns', 'lead gen', 'content marketing'],
        'description': 'Marketing function'
    },
    'RevOps': {
        'keywords': ['revops', 'revenue operations', 'sales ops', 'marketing ops',
                     'operations', 'enablement', 'sales enablement', 'tech stack'],
        'description': 'Revenue Operations'
    },
    'Customer Success': {
        'keywords': ['customer success', 'csm', 'cs team', 'retention', 'churn',
                     'onboarding', 'renewal', 'renewals', 'expansion', 'upsell'],
        'description': 'Customer Success function'
    },
    'Leadership': {
        'keywords': ['cro', 'cmo', 'vp sales', 'vp marketing', 'director', 'head of',
                     'leader', 'leadership', 'executive', 'c-suite', 'founder', 'ceo'],
        'description': 'Executive and leadership roles'
    },
    'Product': {
        'keywords': ['product', 'product manager', 'pm', 'product marketing', 'pmm',
                     'roadmap', 'feature', 'features', 'release', 'launch'],
        'description': 'Product and Product Marketing'
    },
}


def parse_markdown_transcript(file_path: Path) -> dict:
    """Parse a markdown transcript file to extract metadata and content."""
    content = file_path.read_text(encoding='utf-8')

    # Extract episode number from filename
    filename = file_path.stem
    episode_match = re.match(r'(\d+)', filename)
    episode_num = int(episode_match.group(1)) if episode_match else None

    # Extract title from first H1 or filename
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filename

    # Extract date - look for various date patterns
    date = None

    # Pattern 1: "**Date:** Jan 16, 2026" or "Date: Nov 13, 2025"
    abbrev_match = re.search(r'\*?\*?Date:\*?\*?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})', content, re.IGNORECASE)
    if abbrev_match:
        try:
            month_str = abbrev_match.group(1)
            day = abbrev_match.group(2)
            year = abbrev_match.group(3)
            date = datetime.strptime(f"{month_str} {day} {year}", '%b %d %Y')
        except ValueError:
            pass

    # Pattern 2: Full month names "January 16, 2026"
    if date is None:
        full_month_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', content, re.IGNORECASE)
        if full_month_match:
            try:
                month_str = full_month_match.group(1)
                day = full_month_match.group(2)
                year = full_month_match.group(3)
                date = datetime.strptime(f"{month_str} {day} {year}", '%B %d %Y')
            except ValueError:
                pass

    # Pattern 3: ISO format "2024-01-16"
    if date is None:
        iso_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', content)
        if iso_match:
            try:
                date = datetime.strptime(iso_match.group(1), '%Y-%m-%d')
            except ValueError:
                pass

    # Pattern 4: Any ISO date in content
    if date is None:
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
        if iso_match:
            try:
                date = datetime.strptime(iso_match.group(1), '%Y-%m-%d')
            except ValueError:
                pass

    # If no date found, estimate from episode number (starting Sep 2023)
    if date is None and episode_num:
        base_date = datetime(2023, 9, 1)
        estimated_days = (episode_num - 1) * 7
        date = base_date + pd.Timedelta(days=estimated_days)

    # Clean the transcript text
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
    transcript_text = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
    transcript_text = re.sub(r'^[A-Za-z]+:\s*', '', transcript_text, flags=re.MULTILINE)
    transcript_text = re.sub(r'!\[.*?\]\(.*?\)', '', transcript_text)
    transcript_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', transcript_text)
    transcript_text = re.sub(r'\n{3,}', '\n\n', transcript_text)
    transcript_text = transcript_text.strip()

    return {
        'episode_num': episode_num,
        'title': title,
        'date': date,
        'file_path': str(file_path),
        'transcript': transcript_text,
        'word_count': len(transcript_text.split())
    }


def segment_transcript(text: str, min_words: int = MIN_CHUNK_WORDS,
                       max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split transcript into semantic chunks based on paragraphs."""
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_words = len(para.split())

        if para_words > max_words:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0

            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_words = len(sent.split())
                if current_word_count + sent_words > max_words and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sent]
                    current_word_count = sent_words
                else:
                    current_chunk.append(sent)
                    current_word_count += sent_words
        else:
            if current_word_count + para_words > max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_word_count = para_words
            else:
                current_chunk.append(para)
                current_word_count += para_words

    if current_chunk and current_word_count >= min_words:
        chunks.append(' '.join(current_chunk))

    return chunks


def ingest_transcripts() -> list[dict]:
    """Load and parse all transcript files."""
    print("Ingesting transcripts...")
    transcripts = []

    md_files = sorted(TRANSCRIPTS_DIR.glob("*.md"))
    print(f"Found {len(md_files)} transcript files")

    for file_path in md_files:
        try:
            transcript = parse_markdown_transcript(file_path)
            transcripts.append(transcript)
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")

    transcripts.sort(key=lambda x: x['episode_num'] or 0)
    print(f"Successfully parsed {len(transcripts)} transcripts")

    return transcripts


def create_chunks(transcripts: list[dict]) -> tuple[list[str], list[dict]]:
    """Segment all transcripts into chunks with metadata."""
    print("Segmenting transcripts into chunks...")

    all_chunks = []
    chunk_metadata = []

    for transcript in transcripts:
        chunks = segment_transcript(transcript['transcript'])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'episode_num': transcript['episode_num'],
                'title': transcript['title'],
                'date': transcript['date'],
                'chunk_index': i
            })

    print(f"Created {len(all_chunks)} semantic chunks")
    return all_chunks, chunk_metadata


def classify_chunk(chunk: str, taxonomy: dict) -> dict[str, int]:
    """Classify a chunk against a taxonomy, returning match counts per category."""
    chunk_lower = chunk.lower()
    matches = {}

    for category, config in taxonomy.items():
        count = 0
        for keyword in config['keywords']:
            # Use word boundary matching for single words, substring for phrases
            if ' ' in keyword:
                count += chunk_lower.count(keyword)
            else:
                # Match whole words only
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', chunk_lower))
        matches[category] = count

    return matches


def analyze_taxonomy(chunks: list[str], chunk_metadata: list[dict],
                     taxonomy: dict, taxonomy_name: str) -> dict:
    """Analyze all chunks against a taxonomy."""
    print(f"Analyzing {taxonomy_name}...")

    # Initialize counters
    category_counts = defaultdict(int)
    category_chunks = defaultdict(list)
    temporal_data = defaultdict(lambda: defaultdict(int))

    for i, chunk in enumerate(chunks):
        matches = classify_chunk(chunk, taxonomy)
        meta = chunk_metadata[i]

        # Get primary category (highest match count > 0)
        best_category = None
        best_count = 0
        for cat, count in matches.items():
            if count > best_count:
                best_count = count
                best_category = cat

        if best_category and best_count > 0:
            category_counts[best_category] += 1
            category_chunks[best_category].append(chunk)

            # Track temporal data
            if meta['date']:
                quarter = pd.Timestamp(meta['date']).to_period('Q')
                temporal_data[best_category][str(quarter)] += 1

    # Calculate percentages
    total = sum(category_counts.values())
    category_pct = {cat: (count / total * 100) if total > 0 else 0
                    for cat, count in category_counts.items()}

    # Calculate trends
    trends = {}
    for category in taxonomy.keys():
        quarters = sorted(temporal_data[category].keys())
        if len(quarters) >= 4:
            first_half = sum(temporal_data[category][q] for q in quarters[:len(quarters)//2])
            second_half = sum(temporal_data[category][q] for q in quarters[len(quarters)//2:])
            if first_half > 0:
                change_pct = ((second_half - first_half) / first_half) * 100
                if change_pct > 20:
                    trends[category] = 'Rising'
                elif change_pct < -20:
                    trends[category] = 'Falling'
                else:
                    trends[category] = 'Stable'
            else:
                trends[category] = 'Rising' if second_half > 0 else 'Stable'
        else:
            trends[category] = 'Stable'

    return {
        'counts': dict(category_counts),
        'percentages': category_pct,
        'temporal': {cat: dict(data) for cat, data in temporal_data.items()},
        'trends': trends,
        'total_classified': total
    }


def create_taxonomy_chart(analysis: dict, taxonomy: dict, taxonomy_name: str,
                          filename: str, color_palette: str = 'Blues_d') -> None:
    """Create a horizontal bar chart for a taxonomy analysis."""

    # Sort categories by percentage
    sorted_cats = sorted(analysis['percentages'].items(), key=lambda x: x[1], reverse=True)

    # Filter out categories with 0%
    sorted_cats = [(cat, pct) for cat, pct in sorted_cats if pct > 0]

    if not sorted_cats:
        print(f"No data to chart for {taxonomy_name}")
        return

    categories, percentages = zip(*sorted_cats)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.6)))

    # Create color palette - use distinct colors
    if 'Blues' in color_palette:
        colors = sns.color_palette('tab10', len(categories))
    elif 'Greens' in color_palette:
        colors = sns.color_palette('Set2', len(categories))
    else:
        colors = sns.color_palette('husl', len(categories))

    # Create horizontal bar chart
    y_pos = range(len(categories))
    bars = ax.barh(y_pos, percentages, color=colors)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Share of Discussion (%)', fontsize=12)
    ax.set_title(f'GTM AI Podcast: {taxonomy_name}', fontsize=14, fontweight='bold')

    # Add value labels and trend indicators
    for i, (bar, cat) in enumerate(zip(bars, categories)):
        trend = analysis['trends'].get(cat, 'Stable')
        trend_icon = {'Rising': '↑', 'Falling': '↓', 'Stable': '→'}.get(trend, '')
        pct = percentages[i]
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% {trend_icon}', va='center', fontsize=10)

    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {filename}")


def create_trend_chart(tools_analysis: dict, roles_analysis: dict) -> None:
    """Create a combined trend chart showing top categories over time as percentages."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Helper to convert counts to percentages per quarter
    def get_quarterly_percentages(analysis: dict, categories: list) -> dict:
        # Get all quarters and total counts per quarter
        quarter_totals = defaultdict(int)
        for cat in analysis['temporal']:
            for q, count in analysis['temporal'][cat].items():
                quarter_totals[q] += count

        # Calculate percentages
        result = {}
        for cat in categories:
            temporal = analysis['temporal'].get(cat, {})
            result[cat] = {}
            for q in quarter_totals:
                if quarter_totals[q] > 0:
                    result[cat][q] = (temporal.get(q, 0) / quarter_totals[q]) * 100
                else:
                    result[cat][q] = 0
        return result

    # Tools & Tactics trends
    ax1 = axes[0]
    top_tools = sorted(tools_analysis['percentages'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_tool_cats = [cat for cat, _ in top_tools]
    tools_pct = get_quarterly_percentages(tools_analysis, top_tool_cats)
    colors = sns.color_palette('tab10', len(top_tools))  # Distinct colors

    for i, cat in enumerate(top_tool_cats):
        pct_data = tools_pct[cat]
        if pct_data:
            quarters = sorted(pct_data.keys())
            values = [pct_data[q] for q in quarters]
            ax1.plot(quarters, values, marker='o', label=cat, color=colors[i], linewidth=2)

    ax1.set_title('Tools & Tactics Trends', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Share of Discussion (%)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Functions & Roles trends
    ax2 = axes[1]
    top_roles = sorted(roles_analysis['percentages'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_role_cats = [cat for cat, _ in top_roles]
    roles_pct = get_quarterly_percentages(roles_analysis, top_role_cats)
    colors = sns.color_palette('Set2', len(top_roles))  # Distinct colors

    for i, cat in enumerate(top_role_cats):
        pct_data = roles_pct[cat]
        if pct_data:
            quarters = sorted(pct_data.keys())
            values = [pct_data[q] for q in quarters]
            ax2.plot(quarters, values, marker='o', label=cat, color=colors[i], linewidth=2)

    ax2.set_title('Functions & Roles Trends', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Share of Discussion (%)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'taxonomy_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved taxonomy_trends.png")


def generate_report(tools_analysis: dict, roles_analysis: dict,
                    transcripts: list[dict]) -> None:
    """Generate markdown report."""

    report = f"""# GTM AI Podcast Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Episodes Analyzed:** {len(transcripts)}
**Date Range:** {min(t['date'] for t in transcripts if t['date']).strftime('%B %Y')} - {max(t['date'] for t in transcripts if t['date']).strftime('%B %Y')}

---

## Tools & Tactics

What AI tools, technologies, and tactical approaches are discussed.

| Category | Share | Trend |
|----------|-------|-------|
"""

    sorted_tools = sorted(tools_analysis['percentages'].items(), key=lambda x: x[1], reverse=True)
    for cat, pct in sorted_tools:
        if pct > 0:
            trend = tools_analysis['trends'].get(cat, 'Stable')
            trend_icon = {'Rising': '📈', 'Falling': '📉', 'Stable': '➡️'}.get(trend, '')
            report += f"| {cat} | {pct:.1f}% | {trend_icon} {trend} |\n"

    report += f"""
**Total segments classified:** {tools_analysis['total_classified']}

---

## Functions & Roles

What business functions and roles are discussed.

| Category | Share | Trend |
|----------|-------|-------|
"""

    sorted_roles = sorted(roles_analysis['percentages'].items(), key=lambda x: x[1], reverse=True)
    for cat, pct in sorted_roles:
        if pct > 0:
            trend = roles_analysis['trends'].get(cat, 'Stable')
            trend_icon = {'Rising': '📈', 'Falling': '📉', 'Stable': '➡️'}.get(trend, '')
            report += f"| {cat} | {pct:.1f}% | {trend_icon} {trend} |\n"

    report += f"""
**Total segments classified:** {roles_analysis['total_classified']}

---

## Visualizations

- `tools_tactics.png` - Tools & Tactics distribution
- `functions_roles.png` - Functions & Roles distribution
- `taxonomy_trends.png` - Trends over time

---

## Methodology

1. **Segmentation:** Transcripts split into ~200-word semantic chunks
2. **Classification:** Each chunk classified against predefined taxonomies using keyword matching
3. **Aggregation:** Counts aggregated to calculate share percentages
4. **Trend Analysis:** Quarter-over-quarter comparison to detect rising/falling topics

"""

    output_path = OUTPUT_DIR / 'theme_report.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Saved report to {output_path}")


def generate_json_output(tools_analysis: dict, roles_analysis: dict,
                         transcripts: list[dict]) -> None:
    """Generate JSON output."""

    output = {
        'generated_at': datetime.now().isoformat(),
        'total_episodes': len(transcripts),
        'tools_tactics': {
            'categories': [
                {
                    'name': cat,
                    'percentage': tools_analysis['percentages'].get(cat, 0),
                    'count': tools_analysis['counts'].get(cat, 0),
                    'trend': tools_analysis['trends'].get(cat, 'Stable'),
                    'description': TOOLS_TACTICS_TAXONOMY[cat]['description'],
                    'temporal': tools_analysis['temporal'].get(cat, {})
                }
                for cat in TOOLS_TACTICS_TAXONOMY.keys()
            ],
            'total_classified': tools_analysis['total_classified']
        },
        'functions_roles': {
            'categories': [
                {
                    'name': cat,
                    'percentage': roles_analysis['percentages'].get(cat, 0),
                    'count': roles_analysis['counts'].get(cat, 0),
                    'trend': roles_analysis['trends'].get(cat, 'Stable'),
                    'description': FUNCTIONS_ROLES_TAXONOMY[cat]['description'],
                    'temporal': roles_analysis['temporal'].get(cat, {})
                }
                for cat in FUNCTIONS_ROLES_TAXONOMY.keys()
            ],
            'total_classified': roles_analysis['total_classified']
        }
    }

    output_path = OUTPUT_DIR / 'theme_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON to {output_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("GTM AI Podcast Theme Analysis")
    print("Tools & Tactics | Functions & Roles")
    print("=" * 60)
    print()

    # Step 1: Ingest transcripts
    transcripts = ingest_transcripts()

    if not transcripts:
        print("ERROR: No transcripts found!")
        return

    # Step 2: Create semantic chunks
    chunks, chunk_metadata = create_chunks(transcripts)

    # Step 3: Analyze against taxonomies
    tools_analysis = analyze_taxonomy(chunks, chunk_metadata,
                                       TOOLS_TACTICS_TAXONOMY, "Tools & Tactics")
    roles_analysis = analyze_taxonomy(chunks, chunk_metadata,
                                       FUNCTIONS_ROLES_TAXONOMY, "Functions & Roles")

    # Step 4: Generate visualizations
    print("\nCreating visualizations...")
    create_taxonomy_chart(tools_analysis, TOOLS_TACTICS_TAXONOMY,
                          "Tools & Tactics", "tools_tactics.png", "Blues_d")
    create_taxonomy_chart(roles_analysis, FUNCTIONS_ROLES_TAXONOMY,
                          "Functions & Roles", "functions_roles.png", "Greens_d")
    create_trend_chart(tools_analysis, roles_analysis)

    # Step 5: Generate outputs
    generate_report(tools_analysis, roles_analysis, transcripts)
    generate_json_output(tools_analysis, roles_analysis, transcripts)

    print()
    print("=" * 60)
    print("Analysis complete!")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
