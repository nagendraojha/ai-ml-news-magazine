# github_finder.py

import os
import time
import logging
import requests
import re
from typing import List, Dict, Any
from utils.ollama_client import OllamaClient


class GitHubRepoFinder:
    """
    Finds GitHub repositories related to the magazine's lead story.

    - Extracts 3–5 concise search terms with a local LLM (fallback: heuristic).
    - Queries GitHub Search API for each term (stars desc) with optional GITHUB_TOKEN.
    - Returns a deduplicated list of repos with fields used by the template.
    """

    def __init__(self):
        self.ollama = OllamaClient()
        self.logger = logging.getLogger(__name__)
        self.token = os.getenv("GITHUB_TOKEN", "").strip()
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-ML-Magazine-App",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def find_related_repos(self, article: Dict) -> List[Dict]:
        headline = article.get("headline") or article.get("title") or ""
        summary = article.get("summary") or article.get("description") or ""
        self.logger.info(f"🔍 Searching GitHub repos for: {headline[:80]}...")
        terms = self._extract_search_terms(headline, summary)
        if not terms:
            self.logger.warning("No search terms extracted; using fallback repositories.")
            return self._get_fallback_repos()

        repos: List[Dict[str, Any]] = []
        seen = set()
        for term in terms[:3]:
            try:
                results = self._search_github(term)
                for r in results:
                    key = r.get("full_name")
                    if key and key not in seen:
                        seen.add(key)
                        repos.append(r)
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"GitHub search failed for '{term}': {e}")
                continue

        if not repos:
            self.logger.info("No repos found from search; falling back to curated list.")
            return self._get_fallback_repos()

        self.logger.info(f"✅ Collected {len(repos)} related repositories.")
        return repos

    # ---------- Internals ----------

    def _extract_search_terms(self, headline: str, summary: str) -> List[str]:
        text = (headline.strip() + "\n\n" + summary.strip()).strip()
        if not text:
            return []

        prompt = f"""
Extract 3-5 concise GitHub search terms for this AI/ML topic.
Return a single comma-separated line, no quotes, no extra text.

Text:
{ text[:800] }
"""
        try:
            out = self.ollama.generate(
                prompt=prompt,
                model=None,          # uses Config.MAIN_MODEL internally
                temperature=0.2,
                max_tokens=60
            )
            terms = [t.strip() for t in out.split(",") if t.strip()]
            return self._clean_terms(terms)[:5] or self._fallback_terms(headline, summary)
        except Exception:
            return self._fallback_terms(headline, summary)

    def _fallback_terms(self, headline: str, summary: str) -> List[str]:
        base = f"{headline} {summary}".lower()
        words = re.findall(r"[a-zA-Z][a-zA-Z\-\+]{2,}", base)[:40]
        seeds = ["ai", "machine learning", "deep learning", "nlp", "computer vision", "llm", "transformer"]
        uniq = []
        for w in words + seeds:
            if w not in uniq:
                uniq.append(w)
        return self._clean_terms(uniq)[:5]

    def _clean_terms(self, terms: List[str]) -> List[str]:
        out = []
        for t in terms:
            t = t.strip().lower()
            t = re.sub(r"\s+", " ", t)
            t = t.replace("#", "").strip()
            if t and t not in out:
                out.append(t)
        return out

    def _search_github(self, term: str) -> List[Dict[str, Any]]:
        q = f'{term} AI OR "machine learning" in:name,description,readme'
        params = {
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": 5,
        }
        url = "https://api.github.com/search/repositories"
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", []) or []
        repos: List[Dict[str, Any]] = []
        for it in items:
            repos.append({
                "name": it.get("name"),
                "full_name": it.get("full_name"),
                "description": it.get("description") or "",
                "url": it.get("html_url"),
                "stars": it.get("stargazers_count") or 0,
                "forks": it.get("forks_count") or 0,
                "language": it.get("language") or "",
                "topics": it.get("topics") or [],
            })
        return repos

    def _get_fallback_repos(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "scikit-learn",
                "full_name": "scikit-learn/scikit-learn",
                "description": "Machine Learning in Python",
                "url": "https://github.com/scikit-learn/scikit-learn",
                "stars": 58000,
                "forks": 26000,
                "language": "Python",
                "topics": ["machine-learning", "python", "data-science"],
            },
            {
                "name": "pytorch",
                "full_name": "pytorch/pytorch",
                "description": "Tensors and dynamic neural networks in Python",
                "url": "https://github.com/pytorch/pytorch",
                "stars": 75000,
                "forks": 20000,
                "language": "Python",
                "topics": ["deep-learning", "python", "neural-networks"],
            },
            {
                "name": "keras",
                "full_name": "keras-team/keras",
                "description": "Deep Learning for humans",
                "url": "https://github.com/keras-team/keras",
                "stars": 60000,
                "forks": 19000,
                "language": "Python",
                "topics": ["deep-learning", "keras", "neural-networks"],
            },
        ]
