import logging
from typing import List, Dict, Any
from utils.config import Config
from utils.ollama_client import OllamaClient

class HeadlineGenerator:
    def __init__(self):
        self.config = Config
        self.ollama = OllamaClient()
        self.logger = logging.getLogger(__name__)
        self.model = "llama3.2:1b"
    
    def generate_headline(self, article: Dict) -> str:
        """Generate compelling headline for article"""
        title = article.get('title', '')
        description = article.get('description', '')
        
        prompt = f"""
        Create a compelling, professional headline for this AI/ML news article.
        The headline should be engaging, concise (under 15 words), and capture the essence.
        
        Original Title: {title}
        Description: {description}
        
        Generate ONLY the headline without any additional text.
        """
        
        headline = self.ollama.generate(prompt, self.config.FAST_MODEL, temperature=0.8)
        headline = headline.strip().strip('"').strip("'")
        
        # Ensure it's not too long
        if len(headline) > 100:
            words = headline.split()[:15]
            headline = ' '.join(words)
            
        return headline or title