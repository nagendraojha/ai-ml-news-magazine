# import requests
# import json
# import logging
# from typing import Dict, Any, List
# from utils.config import Config

# class OllamaClient:
#     def __init__(self):
#         self.base_url = Config.OLLAMA_BASE_URL
#         self.logger = logging.getLogger(__name__)
    
#     def generate(self, prompt: str, model: str = None, system: str = None, 
#                 temperature: float = 0.7, max_tokens: int = 4000) -> str:
#         """Generate response using Ollama"""
#         if model is None:
#             model = Config.MAIN_MODEL
            
#         payload = {
#             "model": model,
#             "prompt": prompt,
#             "system": system,
#             "options": {
#                 "temperature": temperature,
#                 "num_predict": max_tokens
#             },
#             "stream": False
#         }
        
#         try:
#             response = requests.post(
#                 f"{self.base_url}/api/generate",
#                 json=payload,
#                 timeout=1000
#             )
#             response.raise_for_status()
#             return response.json()["response"]
#         except Exception as e:
#             self.logger.error(f"Ollama API error: {e}")
#             # Fallback to faster model
#             if model != Config.FAST_MODEL:
#                 return self.generate(prompt, Config.FAST_MODEL, system, temperature, max_tokens)
#             return ""
    
#     def generate_structured(self, prompt: str, schema: Dict, model: str = None) -> Dict:
#         """Generate structured JSON response"""
#         system_prompt = f"""You must respond with valid JSON matching this schema: {json.dumps(schema)}
#         Ensure the response is parseable and complete."""
        
#         response = self.generate(prompt, model, system_prompt, temperature=0.3)
        
#         try:
#             # Extract JSON from response if it contains extra text
#             start_idx = response.find('{')
#             end_idx = response.rfind('}') + 1
#             if start_idx != -1 and end_idx != 0:
#                 json_str = response[start_idx:end_idx]
#                 return json.loads(json_str)
#         except:
#             pass
            
#         return {}





# utils/ollama_client.py

# utils/ollama_client.py
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Generator
import requests

from utils.config import Config


class OllamaClient:
    """
    Robust Ollama client:
      - text generation via /api/generate (with fallback to FAST_MODEL)
      - streaming generation via /api/generate (stream=True)
      - embeddings: try several endpoints / payload keys; if unavailable,
        provide a deterministic fallback embedding so pipelines don't fail.
    """

    def __init__(self):
        self.base_url: str = getattr(Config, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.logger = logging.getLogger(__name__)
        self.config = Config

        # Reasonable network timeouts
        self._gen_timeout_s = getattr(self.config, "OLLAMA_GEN_TIMEOUT", 180)
        self._embed_timeout_s = getattr(self.config, "OLLAMA_EMBED_TIMEOUT", 60)

    # -------------------------
    # Text generation (non-streaming)
    # -------------------------
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion.
        Falls back to FAST_MODEL on transport/server error.
        """
        if model is None:
            model = getattr(self.config, "MAIN_MODEL", None) or getattr(self.config, "FAST_MODEL", None)

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        options.update(kwargs or {})

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "options": options,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self._gen_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama variants use "response" for text
            return (data.get("response") or "").strip()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama /api/generate failed: {e}")
            # Fallback to FAST_MODEL if not already using it
            fast_model = getattr(self.config, "FAST_MODEL", None)
            if fast_model and model != fast_model:
                self.logger.warning(f"Falling back to FAST_MODEL: {fast_model}")
                return self.generate(
                    prompt=prompt,
                    model=fast_model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            return ""

    # -------------------------
    # Text generation (streaming)
    # -------------------------
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Stream tokens incrementally.
        Yields text chunks as they arrive; stops when server signals done.
        """
        if model is None:
            model = getattr(self.config, "MAIN_MODEL", None) or getattr(self.config, "FAST_MODEL", None)

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        options.update(kwargs or {})

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "options": options,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self._gen_timeout_s,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                # iterate lines and parse JSON per-line
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        # Some responses may be plain JSON lines, some may be bytes
                        data = json.loads(line)
                    except Exception:
                        # skip unparsable chunk but continue streaming
                        continue
                    chunk = data.get("response") or ""
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama streaming /api/generate failed: {e}")
            return

    # -------------------------
    # Embeddings
    # -------------------------
    def _try_embeddings_endpoints(self, model: str, text: str) -> Optional[List[float]]:
        """
        Try several plausible endpoints / payload key names. Return embedding list
        if successful, otherwise None.
        """
        endpoints = [
            "/api/embeddings",      # common
            "/api/v1/embeddings",   # some variants
            "/embeddings",          # less common
            "/v1/embeddings",       # possibility
        ]
        # payload key variations
        payload_keys = ["prompt", "input", "text"]

        headers = {"Content-Type": "application/json"}

        for ep in endpoints:
            url = self.base_url.rstrip("/") + ep
            for key in payload_keys:
                payload = {"model": model, key: text}
                try:
                    resp = requests.post(url, json=payload, timeout=self._embed_timeout_s, headers=headers)
                except requests.exceptions.RequestException as e:
                    # Network-level error -> continue trying other endpoints
                    self.logger.debug(f"Request to {url} failed: {e}")
                    continue

                if resp.status_code == 404:
                    # endpoint not found: try the next one
                    self.logger.debug(f"Endpoint {url} returned 404")
                    continue

                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    self.logger.debug(f"Non-OK status from {url}: {resp.status_code} / {e}")
                    # Try to parse body - maybe the server provides `data` form
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                else:
                    # Status OK - parse JSON
                    try:
                        data = resp.json()
                    except Exception:
                        data = None

                if not data:
                    continue

                # Common responses:
                # 1) {"embedding": [ ... ] }
                # 2) {"data": [{"embedding": [...]}], ...}
                emb = data.get("embedding")
                if isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb):
                    return emb

                data_arr = data.get("data") or []
                if data_arr and isinstance(data_arr, list) and isinstance(data_arr[0], dict):
                    emb2 = data_arr[0].get("embedding")
                    if isinstance(emb2, list) and all(isinstance(x, (int, float)) for x in emb2):
                        return emb2

                # Some servers embed the vector as stringified JSON in "response" key
                resp_text = data.get("response") or data.get("text") or None
                if isinstance(resp_text, str):
                    # try to parse as JSON list
                    try:
                        parsed = json.loads(resp_text)
                        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                            return parsed
                    except Exception:
                        pass

        return None

    def _fallback_embedding(self, text: str, dim: int = 512) -> List[float]:
        """
        Deterministic fallback embedding derived from SHA256 hashing.
        Produces a `dim`-length vector of floats in [-1, 1].
        This is ONLY a fallback to allow pipelines to operate when the real
        embeddings API is unavailable.
        """
        if dim <= 0:
            dim = 512
        # Use repeated hashing to generate enough bytes
        out = []
        seed = text.encode("utf-8")
        i = 0
        while len(out) < dim:
            h = hashlib.sha256(seed + i.to_bytes(2, "big")).digest()
            # split digest into 4-byte unsigned ints -> convert to float in [-1,1]
            for j in range(0, len(h), 4):
                if len(out) >= dim:
                    break
                chunk = h[j : j + 4]
                val = int.from_bytes(chunk, "big", signed=False)
                # map to [-1,1]
                f = (val / 0xFFFFFFFF) * 2.0 - 1.0
                out.append(float(f))
            i += 1

        # Normalize to unit vector
        norm = sum(x * x for x in out) ** 0.5 or 1.0
        out = [x / norm for x in out]
        return out

    def embeddings(self, model: Optional[str], prompt: str) -> List[float]:
        """
        Return a single embedding vector for the given prompt.
        If server doesn't expose embeddings, gracefully fallback to deterministic software embedding.
        """
        if model is None:
            model = getattr(self.config, "EMBED_MODEL", None) or getattr(self.config, "MAIN_MODEL", None)

        # Try known endpoints / payload forms first
        try:
            emb = self._try_embeddings_endpoints(model, prompt)
            if emb:
                return emb
        except Exception as e:
            self.logger.debug(f"Attempt to query embeddings endpoints raised: {e}")

        # If we reached here, embeddings endpoint(s) unavailable or returned unusable data
        self.logger.error(f"Ollama embeddings endpoints not found or unsupported at {self.base_url}. Using fallback embedding.")
        # Try to read preferred embed dim from config
        prefer_dim = getattr(self.config, "EMBED_DIM", None) or 512
        try:
            prefer_dim = int(prefer_dim)
        except Exception:
            prefer_dim = 512

        return self._fallback_embedding(prompt, dim=prefer_dim)

    # Back-compat helper
    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Convenience wrapper around embeddings().
        """
        return self.embeddings(model=model, prompt=text)
