"""
AI-powered content analysis system.

Standalone extraction from Agency OS — the content analysis layer used inside
the Research Agent to evaluate sources before passing them to the Content
Strategist.

Original: agency-os/services/AIOps/AIOps-Assistant-Local/backend/src/content/analyzer.py
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from .extractor import ContentExtractor

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """AI-powered content analysis and insights generation."""

    def __init__(self):
        # Read config directly from environment — no internal settings module needed
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm = self._initialize_llm()
        self.extractor = ContentExtractor()

    def _initialize_llm(self) -> Optional[ChatOpenAI]:
        """Initialize LLM for content analysis. Returns None if key is missing."""
        try:
            if self.openai_api_key:
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=1000,
                    openai_api_key=self.openai_api_key
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for content analysis: {e}")
        return None

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
        Send content to GPT-4o-mini with a structured JSON prompt.

        The prompt asks for key_insights, sentiment, relevance_score,
        content_type, main_topics, quality_score, and a one-sentence summary.
        Metadata (url, domain, analyzed_at, content_length) is merged in after
        parsing. Falls back to _default_analysis on JSON decode or API errors.
        """
        if not self.llm:
            return self._default_analysis(url)

        try:
            analysis_prompt = f"""
            Analyze this content for key insights, relevance, and sentiment:

            Content: {content[:2000]}...

            Provide analysis as JSON:
            {{
                "key_insights": ["insight 1", "insight 2", "insight 3"],
                "sentiment": "positive/negative/neutral",
                "relevance_score": 8.5,
                "content_type": "news/analysis/tutorial/report",
                "main_topics": ["topic1", "topic2"],
                "quality_score": 7.5,
                "summary": "Brief 1-sentence summary"
            }}

            Return only valid JSON.
            """

            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=analysis_prompt)]
            )

            try:
                analysis = json.loads(response.content)

                # Merge metadata that the LLM doesn't know about
                analysis.update({
                    "analyzed_at": datetime.now().isoformat(),
                    "content_length": len(content),
                    "url": url,
                    "domain": urlparse(url).netloc if url else None
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
