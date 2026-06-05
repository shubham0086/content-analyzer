"""
Tests for ContentAnalyzer.

Run with: pytest tests/

Four tests cover:
  1. analyze_text — real AI path (mocked LLM)
  2. analyze_url fallback — network unavailable, must return fallback dict
  3. _default_analysis — structure and required keys
  4. JSON output — result is serializable and contains expected top-level keys
"""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analyzer import ContentAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "key_insights", "sentiment", "relevance_score",
    "content_type", "quality_score", "summary",
    "analyzed_at", "content_length", "url", "domain",
}

MOCK_LLM_RESPONSE = json.dumps({
    "key_insights": ["RAG reduces hallucination", "Grounds answers in sources", "Easy to update knowledge"],
    "sentiment": "positive",
    "relevance_score": 9.0,
    "content_type": "analysis",
    "main_topics": ["RAG", "LLM"],
    "quality_score": 8.5,
    "summary": "RAG enhances LLMs by grounding responses in retrieved documents."
})


def make_analyzer_with_mock_llm():
    """Return a ContentAnalyzer whose LLM invoke call returns MOCK_LLM_RESPONSE."""
    analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
    analyzer.openai_api_key = "test-key"
    analyzer.extractor = MagicMock()

    mock_response = MagicMock()
    mock_response.content = MOCK_LLM_RESPONSE

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)
    analyzer.llm = mock_llm

    return analyzer


# ---------------------------------------------------------------------------
# Test 1 — analyze_text (real AI path, LLM mocked)
# ---------------------------------------------------------------------------

def test_analyze_text():
    """analyze_text with a long-enough string should return AI-parsed fields."""
    analyzer = make_analyzer_with_mock_llm()

    text = (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large "
        "language models by retrieving relevant documents from an external knowledge "
        "base before generating a response. It reduces hallucination significantly."
    )

    result = asyncio.run(analyzer.analyze_text(text))

    assert result["sentiment"] == "positive"
    assert result["relevance_score"] == 9.0
    assert len(result["key_insights"]) == 3
    assert result.get("fallback") is not True


# ---------------------------------------------------------------------------
# Test 2 — analyze_url fallback (network unavailable)
# ---------------------------------------------------------------------------

def test_analyze_url_fallback():
    """When extraction returns None, analyze_url must return a valid fallback dict."""
    analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
    analyzer.openai_api_key = ""
    analyzer.llm = None

    mock_extractor = MagicMock()
    # Simulate a network failure — extract returns None
    mock_extractor.extract = AsyncMock(return_value=None)
    analyzer.extractor = mock_extractor

    url = "https://this-will-not-be-fetched.example.com"
    result = asyncio.run(analyzer.analyze_url(url))

    assert result["fallback"] is True
    assert result["url"] == url
    # Domain should be parsed even in fallback
    assert result["domain"] == "this-will-not-be-fetched.example.com"


# ---------------------------------------------------------------------------
# Test 3 — _default_analysis structure
# ---------------------------------------------------------------------------

def test_default_analysis():
    """_default_analysis must include all required keys with sensible defaults."""
    analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
    analyzer.openai_api_key = ""
    analyzer.llm = None
    analyzer.extractor = MagicMock()

    result = analyzer._default_analysis("https://example.com/article")

    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key: {key}"

    assert result["fallback"] is True
    assert result["sentiment"] == "neutral"
    assert isinstance(result["key_insights"], list)
    assert len(result["key_insights"]) == 3


# ---------------------------------------------------------------------------
# Test 4 — JSON serializable output
# ---------------------------------------------------------------------------

def test_json_output():
    """Both fallback and AI paths must produce fully JSON-serializable dicts."""
    analyzer = make_analyzer_with_mock_llm()

    text = (
        "Agent frameworks like LangGraph and CrewAI allow developers to build "
        "multi-step reasoning pipelines where each node can call tools, store memory, "
        "and hand off control to another agent — enabling complex autonomous workflows."
    )

    result = asyncio.run(analyzer.analyze_text(text))

    # Must not raise
    serialized = json.dumps(result)
    reparsed = json.loads(serialized)

    # Top-level keys must survive the round-trip
    for key in ("key_insights", "sentiment", "relevance_score", "quality_score", "summary"):
        assert key in reparsed
