# agents/extractor_agent.py

import logging
import json
import re
from typing import List, Dict
from datetime import datetime  # <-- THIS IS THE FIX
from utils.config import Config
from utils.ollama_client import OllamaClient

class ExtractorAgent:
    """
    An agent that specializes in reading an aggregator article (like a "Top 5" list)
    and extracting the individual news items from it.
    """

    def __init__(self):
        self.config = Config
        self.ollama = OllamaClient()
        self.logger = logging.getLogger(__name__)
        self.model = "llama3.2:1b"

    def extract_stories(self, article: Dict) -> List[Dict]:
        """
        Takes a single aggregator article and returns a list of individual "virtual" articles.
        """
        title = article.get("title", "")
        content = article.get("content", "")
        source_name = article.get("source", {}).get("name", "Unknown Source")
        original_url = article.get("url", "#")
        
        self.logger.info(f"⚡ Activating Extractor Agent for aggregator article: '{title[:50]}...'")
        
        if not content:
            self.logger.warning("Aggregator article has no content to extract from. Skipping.")
            return []

        prompt = self._build_extraction_prompt(title, content)

        try:
            response = self.ollama.generate(
                prompt=prompt,
                model=self.model,
                temperature=0.2,
                max_tokens=3000
            )
            extracted_data = self._parse_extraction_response(response)
            
            # Convert the extracted data into our standard article format
            virtual_articles = []
            for item in extracted_data:
                virtual_articles.append({
                    'title': item.get('title', 'Extracted Story'),
                    'description': item.get('summary', ''),
                    'content': item.get('summary', ''),
                    'url': original_url,
                    'publishedAt': article.get('publishedAt', datetime.now().isoformat()),
                    'source': {'name': f"{source_name} (Extracted)"},
                })
            
            self.logger.info(f"✅ Extractor Agent found {len(virtual_articles)} individual stories.")
            return virtual_articles

        except Exception as e:
            self.logger.error(f"❌ Extractor Agent failed: {e}")
            return []

    def _build_extraction_prompt(self, title: str, content: str) -> str:
        """Builds a prompt to instruct the LLM to extract stories into a JSON format."""
        return f"""
You are a data extraction specialist. Your task is to read the following news article, which contains multiple news stories, and extract each individual story into a structured JSON list.

ARTICLE TITLE: {title}
ARTICLE CONTENT:
{content[:4000]}

Please identify each distinct news story mentioned. For each one, provide a concise title and a one-sentence summary.

Return your response as a valid JSON list of objects. Each object must have two keys: "title" and "summary".

Example format:
[
  {{
    "title": "First story's title",
    "summary": "A one-sentence summary of the first story."
  }},
  {{
    "title": "Second story's title",
    "summary": "A one-sentence summary of the second story."
  }}
]

Now, extract the stories from the article provided.
"""

    def _parse_extraction_response(self, response: str) -> List[Dict]:
        """Robustly parses the JSON list from the LLM's response."""
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                self.logger.warning("Extractor Agent: LLM did not return a valid JSON list.")
                return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Extractor Agent: Failed to decode JSON from LLM response: {e}")
            return []