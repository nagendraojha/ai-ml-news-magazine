# utils/config.py
import os
from datetime import datetime


class Config:
    """
    Central configuration for models, APIs, dedup thresholds, and paths.
    Environment variables override defaults where present.
    """
    NEWS_API_KEYS = {
        "newsdata": os.getenv("NEWSDATA_API_KEY", "keys"),
        "newsapi": os.getenv("NEWSAPI_API_KEY", "keys"),
        "gnews": os.getenv("GNEWS_API_KEY", "keys"),
    }

    @classmethod
    def validate_api_keys(cls):
        """Validate that all API keys are present and look valid."""
        issues = []
        for key_name, key_value in cls.NEWS_API_KEYS.items():
            if not key_value or key_value == "your_key_here":
                issues.append(f"Missing or invalid {key_name} API key")
            elif len(key_value) < 10:
                issues.append(f"{key_name} API key seems too short")
        if issues:
            raise ValueError(f"API Key issues: {', '.join(issues)}")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Text generation models
    MAIN_MODEL = os.getenv("MAIN_MODEL", "llama3.2:3b")
    FAST_MODEL = os.getenv("FAST_MODEL", "llama3.2:1b")

    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_ai_news.index")
    FAISS_IDMAP_PATH = os.getenv("FAISS_IDMAP_PATH", "data/faiss_ai_news.idmap.json")

    # Similarity thresholds (tune on validation slice)
    DEDUP_SIM_THRESHOLD = float(os.getenv("DEDUP_SIM_THRESHOLD", "0.84"))
    DEDUP_BORDERLINE_LOW = float(os.getenv("DEDUP_BORDERLINE_LOW", "0.78"))
    DEDUP_BORDERLINE_HIGH = float(os.getenv("DEDUP_BORDERLINE_HIGH", "0.84"))


    DEDUP_TIME_WINDOW_SECONDS = int(os.getenv("DEDUP_TIME_WINDOW_SECONDS", str(7 * 24 * 3600)))

    # LSH/MinHash parameters (candidate blocking)
    LSH_ENABLED = os.getenv("LSH_ENABLED", "1") == "1"
    LSH_THRESHOLD = float(os.getenv("LSH_THRESHOLD", "0.5"))
    LSH_NUM_PERM = int(os.getenv("LSH_NUM_PERM", "128"))
    LSH_SHINGLE_N = int(os.getenv("LSH_SHINGLE_N", "3"))

    # SimHash quick prefilter
    SIMHASH_ENABLED = os.getenv("SIMHASH_ENABLED", "1") == "1"

    VERIFICATION_ITERATIONS = int(os.getenv("VERIFICATION_ITERATIONS", "2"))


    MAGAZINE_TITLE = os.getenv("MAGAZINE_TITLE", "AI/ML Daily Digest")
    CURATOR_NAME = os.getenv("CURATOR_NAME", "Cosmic News AI")

    @staticmethod
    def get_current_date(fmt: str = "%Y-%m-%d") -> str:
        return datetime.now().strftime(fmt)
