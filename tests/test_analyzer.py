"""
Tests for ContentAnalyzer — no LangChain, direct httpx calls.

Run with: pytest tests/
"""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analyzer import ContentAnalyzer

REQUIRED_KEYS = {
    "key_insights", "sentiment", "relevance_score",
    "content_type", "quality_score", "summary",
    "analyzed_at", "content_length", "url", "domain",
}

MOCK_AI_RESPONSE = {
    "key_insights": ["RAG reduces hallucination", "Grounds answers in sources", "Easy to update knowledge"],
    "sentiment": "positive",
    "relevance_score": 9.0,
    "content_type": "analysis",
    "main_topics": ["RAG", "LLM"],
    "quality_score": 8.5,
    "summary": "RAG enhances LLMs by grounding responses in retrieved documents.",
}


def _mock_httpx_response(body: dict):
    """Build a mock httpx response that returns body from .json()."""
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(body)}}]
    }
    return mock_resp


def test_analyze_text():
    """analyze_text calls httpx, parses JSON, returns AI fields."""
    analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
    analyzer.openai_api_key = "sk-test"
    analyzer.extractor = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_httpx_response(MOCK_AI_RESPONSE))

    text = (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large "
        "language models by retrieving relevant documents from an external knowledge "
        "base before generating a response. It reduces hallucination significantly."
    )

    with patch("src.analyzer.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(analyzer.analyze_text(text))

    assert result["sentiment"] == "positive"
    assert result["relevance_score"] == 9.0
    assert len(result["key_insights"]) == 3
    assert result.get("fallback") is not True


def test_analyze_url_fallback():
    """When extraction returns None, returns fallback dict without calling httpx."""
    analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
    analyzer.openai_api_key = ""
    analyzer.extractor = MagicMock()
    analyzer.extractor.extract = AsyncMock(return_value=None)

    url = "https://this-will-not-be-fetched.example.com"
    result = asyncio.run(analyzer.analyze_url(url))

    assert result["fallback"] is True
    assert result["url"] == url
    assert result["domain"] == "this-will-not-be-fetched.example.com"


def test_default_analysis():
    """_default_analysis returns all required keys with correct defaults."""
    analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
    analyzer.openai_api_key = ""
    analyzer.extractor = MagicMock()

    result = analyzer._default_analysis("https://example.com/article")

    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key: {key}"
    assert result["fallback"] is True
    assert result["sentiment"] == "neutral"
    assert len(result["key_insights"]) == 3


def test_json_output():
    """Result is fully JSON-serializable and contains expected keys."""
    analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
    analyzer.openai_api_key = "sk-test"
    analyzer.extractor = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_httpx_response(MOCK_AI_RESPONSE))

    text = (
        "Agent frameworks allow developers to build multi-step reasoning pipelines "
        "where each node can call tools, store memory, and hand off control to "
        "another agent, enabling complex autonomous workflows."
    )

    with patch("src.analyzer.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(analyzer.analyze_text(text))

    serialized = json.dumps(result)
    reparsed = json.loads(serialized)
    for key in ("key_insights", "sentiment", "relevance_score", "quality_score", "summary"):
        assert key in reparsed
