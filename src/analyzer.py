"""
AI-powered content analysis system.

Standalone extraction from Agency OS: the content analysis layer used inside
the Research Agent to evaluate sources before passing them to the Content
Strategist.

Original: agency-os/services/AIOps/AIOps-Assistant-Local/backend/src/content/analyzer.py

No LangChain. Uses direct httpx calls to the OpenAI-compatible endpoint.
Set OPENAI_API_KEY for analysis. Without a key, returns fallback results.
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from .extractor import ContentExtractor

logger = logging.getLogger(__name__)

_OPENAI_URL   = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_TIMEOUT       = 30


class ContentAnalyzer:
    """AI-powered content analysis and insights generation."""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.extractor = ContentExtractor()

    @property
    def _has_llm(self) -> bool:
        return bool(self.openai_api_key)

    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze content from a URL.

        Fetches the page via ContentExtractor, then passes the cleaned text
        to the AI analysis step. Falls back to _default_analysis if either
        step fails.
        """
        try:
            # Extract content from URL using the dedicated extractor module
            content = await self.extractor.extract(url)
            if not content:
                return self._default_analysis(url)

            # Analyze with AI
            analysis = await self._ai_analyze_content(content, url)
            return analysis

        except Exception as e:
            logger.error(f"URL analysis failed for {url}: {e}")
            return self._default_analysis(url)

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze raw text content.

        Skips extraction — text is sent directly to the AI analysis step.
        Falls back gracefully for very short or empty inputs.
        """
        try:
            if not text or len(text.strip()) < 50:
                return self._default_analysis()

            analysis = await self._ai_analyze_content(text)
            return analysis

        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._default_analysis()

    async def _ai_analyze_content(self, content: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Direct httpx call to OpenAI-compatible endpoint. No LangChain.

        Sends content to gpt-4o-mini with a structured JSON prompt.
        Merges url/domain/length metadata into the parsed result.
        Falls back to _default_analysis on any failure.
        """
        if not self._has_llm:
            return self._default_analysis(url)

        prompt = (
            "Analyze this content for key insights, relevance, and sentiment.\n\n"
            f"Content: {content[:2000]}\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            '{"key_insights": ["insight 1", "insight 2", "insight 3"], '
            '"sentiment": "positive/negative/neutral", '
            '"relevance_score": 8.5, '
            '"content_type": "news/analysis/tutorial/report", '
            '"main_topics": ["topic1", "topic2"], '
            '"quality_score": 7.5, '
            '"summary": "Brief 1-sentence summary"}'
        )

        payload = {
            "model": _DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(_OPENAI_URL, headers=headers, json=payload)

            if not resp.is_success:
                logger.error(f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}")
                return self._default_analysis(url)

            raw = resp.json()["choices"][0]["message"]["content"]
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")
            analysis = json.loads(clean)
            analysis.update({
                "analyzed_at": datetime.now().isoformat(),
                "content_length": len(content),
                "url": url,
                "domain": urlparse(url).netloc if url else None,
            })
            return analysis

        except json.JSONDecodeError:
            logger.warning("Failed to parse AI analysis response as JSON")
            return self._default_analysis(url)
        except Exception as e:
            logger.error(f"AI content analysis failed: {e}")
            return self._default_analysis(url)

    def _default_analysis(self, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Fallback result when AI analysis or extraction is unavailable.

        Returned whenever: LLM key is missing, network request fails,
        content extraction returns nothing, or JSON parsing errors out.
        The 'fallback: True' flag lets callers detect this case.
        """
        return {
            "key_insights": [
                "Content requires manual review",
                "Analysis system temporarily unavailable",
                "May contain valuable information"
            ],
            "sentiment": "neutral",
            "relevance_score": 7.0,
            "content_type": "article",
            "main_topics": ["general"],
            "quality_score": 6.0,
            "summary": "Content analysis pending",
            "analyzed_at": datetime.now().isoformat(),
            "content_length": 0,
            "url": url,
            "domain": urlparse(url).netloc if url else None,
            "fallback": True
        }
