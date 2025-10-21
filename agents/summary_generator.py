import logging
from typing import Dict
import requests
from bs4 import BeautifulSoup
from utils.config import Config
from utils.ollama_client import OllamaClient

class SummaryGenerator:
    """
    An intelligent agent that generates comprehensive summaries. If an article's
    content is missing, it will autonomously fetch it from the source URL.
    """

    def __init__(self):
        self.config = Config
        self.ollama = OllamaClient()
        self.logger = logging.getLogger(__name__)
        self.model = "llama3.2:3b"

    def _fetch_full_content(self, url: str) -> str:
        """
        Visits a URL, scrapes the main article text, and returns it.
        This is the agent's new skill.
        """
        if not url:
            return ""
        
        self.logger.info(f"🕸️ Content is missing. Fetching full article from: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # A simple but effective strategy: find the main article body and get all paragraphs
            # This works for most news sites.
            main_content = soup.find('article') or soup.find('main') or soup.body
            if main_content:
                paragraphs = main_content.find_all('p')
                full_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                self.logger.info(f"✅ Successfully extracted {len(full_text)} characters of content.")
                return full_text
            else:
                self.logger.warning("Could not find a main content area to scrape.")
                return ""

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to fetch URL {url}: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"❌ Error parsing content from {url}: {e}")
            return ""

    def generate_comprehensive_summary(self, article: Dict) -> str:
        """
        Generates a full comprehensive summary (400-600 words). If content is
        missing, it fetches it first.
        """
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')

        # --- NEW LOGIC: Check if we need to fetch the content ---
        # If content is empty or looks too short (like just a copy of the description)
        if not content or len(content) < len(description) + 100:
            full_content = self._fetch_full_content(article.get('url'))
            # If fetching was successful, use the new content. Otherwise, fall back to the original.
            if full_content:
                content = full_content

        # If after all that, we still have no content, use the description as a last resort.
        if not content:
            self.logger.warning(f"⚠️ No content available for '{title[:50]}...'. Using description for summary.")
            content = description

        # Build summarization prompt
        prompt = f"""
You are an expert AI/ML news summarizer. Read the following article content and produce a comprehensive summary between 400 and 600 words.

TITLE: {title}
ARTICLE CONTENT:
{content}

Your summary must:
1. Start with the core announcement or development.
2. Provide key technical or business details.
3. Explain the potential impact or significance.
4. Be clear, engaging, and well-written.
"""

        try:
            summary = self.ollama.generate(
                prompt=prompt,
                model=self.model,
                temperature=0.4,
                max_tokens=6000
            )
            return summary.strip()
        except Exception as e:
            self.logger.error(f"❌ Summary generation LLM error: {e}")
            # Fallback if summarization fails
            return f"{description} [Summary generation failed]"
