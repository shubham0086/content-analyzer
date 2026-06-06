# Content Analyzer

> **IMPORTANT**: This repository contains real, production-ready, battle-tested code extracted directly from active commercial systems (like Agency OS or Founder Growth OS), rather than simplified mock learning artifacts.
>
> For project walkthroughs, architecture flowcharts, and system context, visit the live landing page: [shubham0086.github.io/MyPortfolio.github.io/projects/content-analyzer.html](https://shubham0086.github.io/MyPortfolio.github.io/projects/content-analyzer.html)

> Fetch any URL or paste any text. Get back key insights, multi-level summaries, sentiment, and quality score.

Extracted from Agency OS: the content analysis layer used inside the Research Agent to evaluate sources before passing them to the Content Strategist.

## What It Does

Input: a URL or raw text.
Output:
- key_insights (3 bullet points)
- sentiment (positive / negative / neutral)
- relevance_score (0-10)
- content_type (news / analysis / tutorial / report)
- quality_score (0-10)
- summary (one sentence)

## Architecture

- `src/extractor.py`: fetches URLs, strips nav/footer/scripts, extracts main content from article/main selectors, limits to 5000 chars for token efficiency
- `src/analyzer.py`: sends content to GPT-4o-mini with a structured JSON prompt. Parses response. Falls back gracefully if LLM or network unavailable.

## Quick Start

```bash
git clone https://github.com/shubham0086/content-analyzer
cd content-analyzer
pip install -r requirements.txt
cp .env.example .env
# edit .env: add your OPENAI_API_KEY
python demo/run.py
```

## Where This Fits in Agency OS

The Content Analyzer runs inside the Research Agent as the `analyze_content` tool. Each search result URL gets analyzed before being ranked and passed downstream. See [Agency OS](https://github.com/shubham0086/agency-os) for the full pipeline.
