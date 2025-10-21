import logging
import re
from typing import List, Dict, Tuple
import concurrent.futures
import json
from utils.config import Config
from utils.ollama_client import OllamaClient

class ContentFilter:
    """
    Advanced ContentFilter with robust JSON parsing for LLM responses.
    """

    def __init__(self, batch_size: int = 50, max_workers: int = 5):
        self.config = Config
        self.ollama = OllamaClient()
        self.logger = logging.getLogger(__name__)
        self.model = "llama3.2:3b"
        self.batch_size = batch_size
        self.max_workers = max_workers

    def classify_articles(self, articles: List[Dict]) -> List[Tuple[Dict, str]]:
        """
        Comprehensive classification with parallel batch processing.
        """
        if not articles:
            return []

        self.logger.info(f"🎯 Starting parallel AI/ML classification of {len(articles)} articles...")

        # Step 1: Remove empty articles
        articles = [a for a in articles if a.get("title") or a.get("description")]
        self.logger.info(f"🗑 Removed empty articles, {len(articles)} remain.")

        # Step 2: Deduplication
        seen_titles = set()
        unique_articles = []
        for article in articles:
            title = article.get("title", "").strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        self.logger.info(f"🔍 Deduplicated articles, {len(unique_articles)} remain.")

        # Step 3: Parallel LLM-based AI/ML classification
        classified_articles = self._process_articles_in_parallel(unique_articles)
        
        self.logger.info(f"🎊 Parallel classification complete. {len(classified_articles)}/{len(unique_articles)} AI/ML articles found.")
        return classified_articles

    def _process_articles_in_parallel(self, articles: List[Dict]) -> List[Tuple[Dict, str]]:
        """Process articles in parallel batches."""
        all_classified = []
        total_batches = (len(articles) + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"⚡ Processing {len(articles)} articles in {total_batches} batches with {self.max_workers} parallel workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches for parallel processing
            future_to_batch = {}
            
            for batch_num, i in enumerate(range(0, len(articles), self.batch_size), 1):
                batch = articles[i:i + self.batch_size]
                future = executor.submit(self._process_single_batch, batch, batch_num, total_batches)
                future_to_batch[future] = batch_num
            
            # Collect results as they complete
            completed_batches = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=300)  # 5-minute timeout per batch
                    all_classified.extend(batch_results)
                    completed_batches += 1
                    self.logger.info(f"✅ Batch {batch_num}/{total_batches} completed: {len(batch_results)} AI articles found")
                except concurrent.futures.TimeoutError:
                    self.logger.error(f"⏰ Batch {batch_num} timed out after 5 minutes")
                except Exception as e:
                    self.logger.error(f"❌ Batch {batch_num} failed: {e}")
        
        return all_classified

    def _process_single_batch(self, batch: List[Dict], batch_num: int, total_batches: int) -> List[Tuple[Dict, str]]:
        """Process a single batch of articles."""
        batch_results = []
        
        self.logger.info(f"🧠 Processing batch {batch_num}/{total_batches} with {len(batch)} articles...")
        
        for article in batch:
            try:
                classification_result = self._analyze_article_with_llm(article)
                
                if classification_result["is_ai_related"]:
                    batch_results.append((article, classification_result["story_type"]))
                    self.logger.debug(f"✅ AI/ML: '{article.get('title', '')[:60]}...' → {classification_result['story_type']}")
                else:
                    self.logger.debug(f"❌ NOT AI: '{article.get('title', '')[:60]}...'")
                
                # Add confidence and reasoning to article metadata
                article['ai_analysis'] = {
                    'confidence': classification_result.get('confidence', 'unknown'),
                    'reasoning': classification_result.get('reasoning', ''),
                    'is_ai_related': classification_result['is_ai_related']
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to analyze article '{article.get('title', '')[:60]}...': {e}")
                # Mark as non-AI on failure
                article['ai_analysis'] = {
                    'confidence': 'low',
                    'reasoning': f'Analysis failed: {str(e)[:100]}',
                    'is_ai_related': False
                }
        
        self.logger.info(f"📊 Batch {batch_num} complete: {len(batch_results)}/{len(batch)} AI articles")
        return batch_results

    def _analyze_article_with_llm(self, article: Dict) -> Dict:
        """
        Use LLM to comprehensively analyze if the article is AI/ML/Data Science related.
        """
        title = article.get("title", "").strip()
        description = article.get("description", "").strip()
        content = article.get("content", "").strip()
        
        if not any([title, description, content]):
            return {
                "is_ai_related": False,
                "story_type": "NOT_AI",
                "confidence": "low",
                "reasoning": "No content to analyze"
            }

        prompt = f"""
You are an expert AI/ML/Data Science content analyst. Analyze this news article and determine:

1. Is this article primarily about Artificial Intelligence, Machine Learning, or Data Science?
2. If yes, is it about ONE specific development (SINGLE_STORY) or MULTIPLE items (AGGREGATOR_STORY)?

ARTICLE DETAILS:
TITLE: {title}
DESCRIPTION: {description}
CONTENT: {content[:800]}  # Reduced content length for better JSON generation

CRITERIA FOR AI/ML/Data Science:
- Artificial Intelligence (AI) technologies, research, applications
- Machine Learning (ML) models, algorithms, frameworks
- Deep Learning, Neural Networks, Transformers, LLMs
- Data Science, Analytics, Big Data, Predictive Modeling
- AI companies (OpenAI, Anthropic, Google AI, Meta AI, etc.)
- ML frameworks (PyTorch, TensorFlow, scikit-learn, etc.)
- AI applications (computer vision, NLP, robotics, autonomous systems)

RESPONSE FORMAT (JSON only):
{{
    "is_ai_related": true/false,
    "story_type": "SINGLE_STORY" or "AGGREGATOR_STORY" or "NOT_AI",
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation"
}}

IMPORTANT: 
- Be strict but fair. Only mark as AI/ML if it's the PRIMARY focus.
- Keep reasoning SHORT (1-2 sentences max)
- Respond with ONLY the JSON, no other text.
- Ensure proper JSON formatting with correct commas and quotes.
"""

        try:
            response = self.ollama.generate(
                prompt=prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=300  # Reduced tokens for more focused response
            )
            
            # Try to parse JSON response with robust error handling
            analysis_result = self._robust_parse_llm_response(response)
            
            if analysis_result:
                return analysis_result
            else:
                # Fallback analysis if JSON parsing fails
                return self._fallback_analysis(title, description, content)
                
        except Exception as e:
            self.logger.error(f"❌ LLM analysis failed: {e}")
            return self._fallback_analysis(title, description, content)

    def _robust_parse_llm_response(self, response: str) -> Dict:
        """Robust JSON parsing with multiple fallback strategies."""
        if not response:
            return None
            
        # Strategy 1: Direct JSON parsing
        try:
            result = json.loads(response.strip())
            if self._validate_json_structure(result):
                return result
        except json.JSONDecodeError:
            pass  # Try other strategies
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if self._validate_json_structure(result):
                    return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find any JSON-like structure
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if json_match:
            try:
                # Try to fix common JSON issues
                json_str = self._fix_common_json_issues(json_match.group(0))
                result = json.loads(json_str)
                if self._validate_json_structure(result):
                    return result
            except json.JSONDecodeError as e:
                self.logger.debug(f"🔧 JSON repair failed: {e}")
        
        # Strategy 4: Manual extraction from text
        return self._extract_from_text_advanced(response)

    def _validate_json_structure(self, result: Dict) -> bool:
        """Validate that the JSON has the required structure."""
        required_fields = ['is_ai_related', 'story_type', 'confidence', 'reasoning']
        if not all(field in result for field in required_fields):
            return False
        
        # Validate field types
        if not isinstance(result['is_ai_related'], bool):
            return False
            
        valid_story_types = ['SINGLE_STORY', 'AGGREGATOR_STORY', 'NOT_AI']
        if result['story_type'] not in valid_story_types:
            return False
            
        valid_confidences = ['high', 'medium', 'low']
        if result['confidence'] not in valid_confidences:
            return False
            
        return True

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Fix 1: Add missing quotes around keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix 2: Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix 3: Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix 4: Ensure proper boolean values
        json_str = re.sub(r':\s*true\b', ': true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r':\s*false\b', ': false', json_str, flags=re.IGNORECASE)
        
        # Fix 5: Remove extra whitespace
        json_str = re.sub(r'\s+', ' ', json_str)
        
        return json_str.strip()

    def _extract_from_text_advanced(self, response: str) -> Dict:
        """Advanced text extraction with better pattern matching."""
        response_lower = response.lower()
        
        # Extract is_ai_related with better logic
        is_ai_related = self._extract_ai_related_from_text(response_lower)
        
        # Extract story type
        story_type = self._extract_story_type_from_text(response_lower, is_ai_related)
        
        # Extract confidence
        confidence = self._extract_confidence_from_text(response_lower)
        
        # Extract reasoning (first meaningful sentence)
        reasoning = self._extract_reasoning_from_text(response)
        
        return {
            "is_ai_related": is_ai_related,
            "story_type": story_type,
            "confidence": confidence,
            "reasoning": reasoning
        }

    def _extract_ai_related_from_text(self, response_lower: str) -> bool:
        """Extract AI-related flag from text with better logic."""
        positive_indicators = [
            'true', 'yes', 'related', 'ai', 'artificial intelligence', 
            'machine learning', 'data science', 'is about ai', 'involves ai'
        ]
        
        negative_indicators = [
            'false', 'no', 'not ai', 'not related', 'unrelated',
            'not about ai', 'no ai content'
        ]
        
        # Count positive vs negative indicators
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        if positive_count > negative_count:
            return True
        elif negative_count > positive_count:
            return False
        else:
            # Fallback to keyword matching
            ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'data science']
            return any(keyword in response_lower for keyword in ai_keywords)

    def _extract_story_type_from_text(self, response_lower: str, is_ai_related: bool) -> str:
        """Extract story type from text."""
        if not is_ai_related:
            return "NOT_AI"
            
        aggregator_indicators = ['aggregator', 'multiple', 'list', 'roundup', 'summary', 'collection']
        single_indicators = ['single', 'specific', 'particular', 'one story']
        
        aggregator_count = sum(1 for indicator in aggregator_indicators if indicator in response_lower)
        single_count = sum(1 for indicator in single_indicators if indicator in response_lower)
        
        if aggregator_count > single_count:
            return "AGGREGATOR_STORY"
        else:
            return "SINGLE_STORY"

    def _extract_confidence_from_text(self, response_lower: str) -> str:
        """Extract confidence level from text."""
        if 'high' in response_lower:
            return "high"
        elif 'medium' in response_lower:
            return "medium"
        elif 'low' in response_lower:
            return "low"
        else:
            # Default to medium if no confidence indicated
            return "medium"

    def _extract_reasoning_from_text(self, response: str) -> str:
        """Extract reasoning from text response."""
        # Remove JSON-like structures
        cleaned = re.sub(r'\{.*?\}', '', response, flags=re.DOTALL)
        
        # Take first meaningful sentence
        sentences = re.split(r'[.!?]+', cleaned)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.startswith(('{', '}', '[', ']')):
                return sentence[:200]  # Limit length
        
        return "Analysis completed via text extraction"

    def _fallback_analysis(self, title: str, description: str, content: str) -> Dict:
        """Enhanced fallback analysis with better keyword matching."""
        text = f"{title} {description} {content}".lower()
        
        # Enhanced keyword matching
        ai_keywords = [
            'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
            'neural network', 'transformer', 'llm', 'gpt', 'chatgpt', 'openai', 'anthropic',
            'data science', 'data analytics', 'big data', 'pytorch', 'tensorflow', 'hugging face',
            'computer vision', 'natural language processing', 'nlp', 'reinforcement learning',
            'generative ai', 'stable diffusion', 'midjourney', 'dall-e', 'claude', 'gemini'
        ]
        
        is_ai_related = any(keyword in text for keyword in ai_keywords)
        
        # Enhanced story type detection
        aggregator_indicators = ['top', 'list', 'roundup', 'summary', 'multiple', 'this week', 'best of']
        is_aggregator = any(indicator in text for indicator in aggregator_indicators)
        
        story_type = "AGGREGATOR_STORY" if is_aggregator else "SINGLE_STORY"
        
        return {
            "is_ai_related": is_ai_related,
            "story_type": story_type if is_ai_related else "NOT_AI",
            "confidence": "low",
            "reasoning": "Enhanced fallback analysis using keyword matching"
        }

    def get_ai_articles_only(self, articles: List[Dict]) -> List[Dict]:
        """
        Convenience method to get only AI/ML related articles.
        """
        classified = self.classify_articles(articles)
        return [article for article, classification in classified if classification != "NOT_AI"]