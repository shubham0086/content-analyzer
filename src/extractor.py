"""
URL content extraction — fetches a page, strips navigation/scripts/footer,
pulls text from article/main selectors, limits to 5000 chars for token efficiency.

Extracted from ContentAnalyzer._extract_content in Agency OS.
"""
import asyncio
import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Fetches and cleans HTML content from a URL."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            # Mimic a real browser so sites don't block the request outright
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    async def extract(self, url: str) -> Optional[str]:
        """
        Fetch a URL and return cleaned plain text.

        Steps:
          1. GET the page (10 s timeout, runs in thread to keep the event loop free)
          2. Parse with BeautifulSoup
          3. Remove script / style / nav / footer / header noise
          4. Try article/main content selectors before falling back to full body text
          5. Collapse whitespace and cap at 5000 chars (token efficiency)

        Returns None on any network or parse error.
        """
        try:
            response = await asyncio.to_thread(
                self.session.get,
                url,
                timeout=10
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove elements that are never useful for content analysis
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Try semantic selectors first — they usually contain the main body
            content_selectors = [
                'article', 'main', '.content', '#content',
                '.post-content', '.entry-content', '.article-content'
            ]

            content = ""
            for selector in content_selectors:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(strip=True)
                    break

            # Fall back to the entire remaining page text
            if not content:
                content = soup.get_text(strip=True)

            # Collapse runs of whitespace to a single space
            content = ' '.join(content.split())

            # Cap at 5000 chars to keep prompts within a reasonable token budget
            return content[:5000]

        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None
