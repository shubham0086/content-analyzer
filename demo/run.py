"""
Demo: analyze a URL and a raw text block, print structured output.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # add your OPENAI_API_KEY
    python demo/run.py
"""
import asyncio
import json
import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from src.analyzer import ContentAnalyzer

load_dotenv()


SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that enhances large language
models by retrieving relevant documents from an external knowledge base before
generating a response. Unlike pure parametric models, RAG grounds answers in
up-to-date, verifiable sources, reducing hallucination and improving factual
accuracy. It has become the dominant pattern for enterprise AI assistants because
it separates the knowledge store (easy to update) from the model weights (expensive
to retrain).
""".strip()

SAMPLE_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"


async def main():
    analyzer = ContentAnalyzer()

    # --- Text analysis ---
    print("=" * 60)
    print("Analyzing raw text block...")
    print("=" * 60)
    text_result = await analyzer.analyze_text(SAMPLE_TEXT)
    print(json.dumps(text_result, indent=2))

    # --- URL analysis ---
    print()
    print("=" * 60)
    print(f"Analyzing URL: {SAMPLE_URL}")
    print("=" * 60)
    url_result = await analyzer.analyze_url(SAMPLE_URL)
    print(json.dumps(url_result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
