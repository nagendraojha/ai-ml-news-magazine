import logging
import time
import os
import threading
import json
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests

# Optional dependencies
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except Exception:
    HAS_DATASKETCH = False

try:
    import xxhash
    HAS_XXHASH = True
except Exception:
    HAS_XXHASH = False

try:
    from simhash import Simhash
    HAS_SIMHASH = True
except Exception:
    HAS_SIMHASH = False

from utils.config import Config
from utils.ollama_client import OllamaClient


# ------------------------ CONTENT FILTER ------------------------

class ContentFilter:
    """
    AI/ML/Data Science news filter + LLM classifier
    """
    AI_KEYWORDS = [
        # General AI/ML
        "artificial intelligence", "ai", "machine learning", "ml", "deep learning", "dl",
        "neural network", "nn", "reinforcement learning", "rl", "unsupervised learning",
        "supervised learning", "self-supervised learning", "semi-supervised learning",
        # NLP / LLMs
        "natural language processing", "nlp", "transformer", "bert", "gpt", "llm", "chatgpt",
        "llama", "flan-t5", "codex", "gpt-4", "gpt-3", "llama2", "gpt-neo",
        # Computer vision
        "computer vision", "cv", "cnn", "resnet", "gan", "stable diffusion", "vae", "image recognition",
        # Data Science / Analytics
        "data science", "data analytics", "predictive analytics", "big data", "data mining",
        "feature engineering", "data visualization", "statistical modeling", "regression", "classification",
        # Frameworks / Libraries
        "tensorflow", "pytorch", "scikit-learn", "keras", "xgboost", "lightgbm", "pandas", "numpy",
        # Generative / Cutting-edge AI
        "generative ai", "prompt engineering", "multi-modal ai", "reinforcement learning from human feedback",
        "rlhf", "foundation model", "self-driving ai", "ai research", "ai paper", "ai model"
    ]

    def __init__(self, batch_size: int = 10, model: str = "llama3.2:1b"):
        self.config = Config
        self.ollama = OllamaClient()
        self.logger = logging.getLogger("ContentFilter")
        self.model = model
        self.batch_size = batch_size

    def classify_articles(self, articles: List[Dict]) -> List[Tuple[Dict, str]]:
        if not articles:
            return []

        # Remove empty articles
        articles = [a for a in articles if a.get("title") or a.get("description")]
        # Deduplicate by title
        seen = set()
        unique_articles = []
        for a in articles:
            title = (a.get("title") or "").strip().lower()
            if title not in seen:
                seen.add(title)
                unique_articles.append(a)
        # Keyword pre-filter
        filtered_articles = [
            a for a in unique_articles
            if self._is_ai_related(a.get("title", ""), a.get("description", ""))
        ]

        # Batch LLM classification
        classified_articles = []
        for i in range(0, len(filtered_articles), self.batch_size):
            batch = filtered_articles[i:i+self.batch_size]
            for article in batch:
                classification = self._get_article_classification(article)
                classified_articles.append((article, classification))
        return classified_articles

    def _is_ai_related(self, title: str, description: str) -> bool:
        text = f"{title} {description}".lower()
        return any(k in text for k in self.AI_KEYWORDS)

    def _get_article_classification(self, article: Dict) -> str:
        title = article.get("title", "").strip()
        description = article.get("description", "").strip()
        prompt = self._build_classification_prompt(title, description)
        try:
            resp = self.ollama.generate(
                prompt=prompt,
                model=self.model,
                temperature=0.0,
                max_tokens=20
            )
            return self._parse_llm_classification(resp)
        except Exception:
            return "NOT_AI"

    def _build_classification_prompt(self, title: str, description: str) -> str:
        return f"""You are a highly precise news classifier.
Classify this article strictly:

TITLE: {title}
DESCRIPTION: {description}

Respond with ONLY ONE of:
SINGLE_STORY
AGGREGATOR_STORY
NOT_AI"""

    def _parse_llm_classification(self, response: str) -> str:
        resp = (response or "").strip().upper()
        if "AGGREGATOR_STORY" in resp:
            return "AGGREGATOR_STORY"
        elif "SINGLE_STORY" in resp:
            return "SINGLE_STORY"
        else:
            return "NOT_AI"


# ------------------------ DEDUPLICATOR ------------------------

class Deduplicator:
    """
    Scalable deduplicator with:
    - Exact hash
    - SimHash
    - MinHash-LSH
    - FAISS embeddings + cosine similarity
    - LLM arbitration for borderline
    Supports batch embeddings and billion-scale pipelines.
    """

    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        llm_model: str = "qwen3:4b",
        faiss_index_path: Optional[str] = "data/faiss_ai_news.index",
        idmap_path: Optional[str] = "data/faiss_ai_news.idmap.json",
        sim_threshold: float = 0.84,
        borderline_low: float = 0.78,
        borderline_high: float = 0.84,
        batch_size: int = 16,
        use_simhash: bool = True,
        use_lsh: bool = True,
    ):
        self.logger = logging.getLogger("Deduplicator")
        self.config = Config
        self.ollama = OllamaClient()
        self.embed_model = embed_model
        self.llm_model = llm_model

        self.sim_threshold = sim_threshold
        self.borderline_low = borderline_low
        self.borderline_high = borderline_high
        self.batch_size = batch_size

        self.use_simhash = use_simhash and HAS_SIMHASH
        self.use_lsh = use_lsh and HAS_DATASKETCH

        self.faiss_index_path = faiss_index_path
        self.idmap_path = idmap_path

        self.id2meta = {}
        self.meta_lock = threading.Lock()
        self.next_id = 0

        self.lsh = MinHashLSH(threshold=0.5, num_perm=128) if self.use_lsh else None

        self.dim = 768
        self.index = None
        if HAS_FAISS:
            self._init_faiss()

        self.exact_hashes = set()
        self.simhash_buckets = {}

        self._load_state()

    # -------------------- Public API --------------------

    def remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        if len(articles) <= 1:
            return articles

        # Sort by description length
        articles = sorted(articles, key=lambda a: len(a.get('description', '') or ''), reverse=True)

        unique = []
        now = int(time.time())
        batch_texts = [self._embed_payload(a) for a in articles]
        batch_embs = self._batch_embed_texts(batch_texts)

        for art, emb in zip(articles, batch_embs):
            norm_title, norm_desc = self._normalize(art.get('title', '')), self._normalize(art.get('description', ''))
            key_text = (norm_title + " " + norm_desc).strip()
            if not key_text:
                key_text = self._normalize(art.get('title', ''))

            if self._is_exact_duplicate(key_text):
                continue

            if self.use_simhash and self._has_simhash_near_dupe(key_text):
                continue

            # ANN check
            is_dup = False
            chosen_parent = None
            if emb is not None and HAS_FAISS and self.index and self.index.ntotal > 0:
                neighbors = self._ann_neighbors(emb, topk=10)
                best = max(neighbors, key=lambda x: x[1], default=None)
                if best and best[1] >= self.sim_threshold:
                    is_dup = True
                    chosen_parent = best
                elif best and self.borderline_low <= best[1] < self.borderline_high:
                    parent_meta = self.id2meta.get(best[0])
                    parent_art = parent_meta['article'] if parent_meta else None
                    if parent_art and self._llm_same_event(art, parent_art):
                        is_dup = True
                        chosen_parent = best
            elif emb is None:
                # fallback: LLM check with last few entries
                if self._llm_says_duplicate_any(art, list(self.id2meta.keys())[-3:]):
                    continue

            if is_dup:
                continue

            unique.append(art)
            self._index_new_article(art, key_text, emb, now_ts=now)

        self._save_state()
        return unique

    # -------------------- Embedding & ANN --------------------

    def _embed_payload(self, art: Dict) -> str:
        return ((art.get('title') or '').strip() + "\n\n" + (art.get('description') or '').strip())[:2000]

    def _batch_embed_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Batch embeddings for efficiency"""
        results = []
        for t in texts:
            try:
                vec = self.ollama.embeddings(model=self.embed_model, prompt=t)
                results.append(np.array(vec, dtype=np.float32))
            except Exception:
                results.append(None)
        return results

    def _init_faiss(self):
        self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64

    def _norm(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-9)

    def _ann_neighbors(self, vec: np.ndarray, topk: int = 10) -> List[Tuple[int, float]]:
        q = self._norm(vec).reshape(1, -1).astype('float32')
        sims, ids = self.index.search(q, topk)
        out = []
        for i, s in zip(ids[0], sims[0]):
            if i == -1:
                continue
            out.append((int(i), float(s)))
        return out

    # -------------------- LLM arbitration --------------------

    def _llm_same_event(self, a1: Dict, a2: Dict) -> bool:
        prompt = f"""You are a news analyst. Determine if these two articles report the same core event.

Article 1:
TITLE: {a1.get('title')}
SUMMARY: {(a1.get('description') or '')[:300]}

Article 2:
TITLE: {a2.get('title')}
SUMMARY: {(a2.get('description') or '')[:300]}

Respond with YES or NO."""
        try:
            resp = self.ollama.generate(prompt, model=self.llm_model, temperature=0.0, max_tokens=3)
            return (resp or "").strip().upper().startswith("YES")
        except Exception:
            return False

    def _llm_says_duplicate_any(self, a_new: Dict, cand_ids: List[int]) -> bool:
        for cid in cand_ids:
            meta = self.id2meta.get(cid)
            if meta and self._llm_same_event(a_new, meta['article']):
                return True
        return False

    # -------------------- Indexing & persistence --------------------

    def _index_new_article(self, art: Dict, key_text: str, emb: Optional[np.ndarray], now_ts: int):
        with self.meta_lock:
            idx = self.next_id
            self.next_id += 1
            self.id2meta[idx] = {"id": idx, "article": art, "ts": now_ts, "vec": emb}

        if HAS_FAISS and emb is not None:
            self.index.add_with_ids(self._norm(emb).reshape(1, -1).astype('float32'), np.array([idx], dtype=np.int64))

    # -------------------- Utility --------------------

    def _normalize(self, s: Optional[str]) -> str:
        if not s:
            return ""
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _is_exact_duplicate(self, text: str) -> bool:
        h = xxhash.xxh3_64_hexdigest(text.encode('utf-8')) if HAS_XXHASH else str(hash(text))
        if h in self.exact_hashes:
            return True
        self.exact_hashes.add(h)
        return False

    def _has_simhash_near_dupe(self, text: str) -> bool:
        if not HAS_SIMHASH or not self.use_simhash:
            return False
        s = Simhash(text)
        prefix = format(s.value, '064b')[:16]
        bucket = self.simhash_buckets.setdefault(prefix, [])
        if bucket:
            return True
        bucket.append(format(s.value, '064b'))
        return False

    # -------------------- Persistence --------------------

    def _load_state(self):
        # FAISS
        try:
            if HAS_FAISS and self.faiss_index_path and os.path.exists(self.faiss_index_path):
                self.index = faiss.read_index(self.faiss_index_path)
        except Exception:
            pass
        # ID map
        try:
            if self.idmap_path and os.path.exists(self.idmap_path):
                with open(self.idmap_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.id2meta = {int(k): v for k, v in data.get("id2meta", {}).items()}
                if self.id2meta:
                    self.next_id = max(self.id2meta.keys()) + 1
        except Exception:
            pass

    def _save_state(self):
        # FAISS
        try:
            if HAS_FAISS and self.faiss_index_path and self.index:
                os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
                faiss.write_index(self.index, self.faiss_index_path)
        except Exception:
            pass
        # ID map
        try:
            if self.idmap_path:
                os.makedirs(os.path.dirname(self.idmap_path), exist_ok=True)
                slim = {}
                for k, v in self.id2meta.items():
                    slim[k] = {"ts": v.get("ts"), "article": v.get("article")}
                with open(self.idmap_path, "w", encoding="utf-8") as f:
                    json.dump({"id2meta": slim}, f)
        except Exception:
            pass
