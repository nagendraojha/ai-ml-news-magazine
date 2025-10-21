# app.py

import logging
import os
import json
import io  # <-- THIS WAS MISSING
import re
import time
from datetime import datetime
from typing import List, Dict, Any

import requests
from flask import Flask, render_template, jsonify

# Agent Imports
from agents.news_fetcher import NewsFetcher
from agents.content_filter import ContentFilter
from agents.deduplicator import Deduplicator
from agents.extractor_agent import ExtractorAgent
from agents.headline_generator import HeadlineGenerator
from agents.summary_generator import SummaryGenerator
from utils.github_finder import GitHubRepoFinder
from agents.design_engine import DesignEngine
from utils.config import Config

# PDF Generation Imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# --- Main App Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# === LLM DYNAMIC ANALYSIS FUNCTIONS ===

def analyze_article_impacts_with_ollama(article: Dict, model: str = "llama3.2:3b") -> Dict:
    """Use Ollama LLM to generate dynamic impact analysis for each article."""
    headline = article.get('headline', '')
    summary = article.get('summary', '')

    # If no meaningful content, return "no thank you" response
    if not headline and not summary:
        return {
            "user_impact": ["No analysis available for this article."],
            "pros": ["Unable to generate analysis."],
            "cons": ["Content insufficient for evaluation."]
        }
    
    prompt = f"""
You are a senior AI strategist and technology analyst. Your task is to provide a deep and insightful analysis of the following AI/ML news article.

ARTICLE:
TITLE: {headline}
SUMMARY: {summary}

Based on this article, generate a structured analysis in a valid JSON format. The JSON object must contain three keys: "user_impact", "pros", and "cons". Each key should have a list of exactly five distinct, concise, and insightful points (as strings).

GUIDELINES:
- **user_impact**: Focus on how this development will directly affect end-users, developers, consumers, or society.
- **pros**: List the clear advantages, benefits, and positive outcomes of this technology or news.
- **cons**: List the potential risks, disadvantages, negative implications, or challenges.
- Be specific and base your analysis ONLY on the provided article summary. Do not use generic statements.
- Each point in the lists must be a complete sentence.

CRITICAL: Your response MUST be a single, valid JSON object and nothing else. No additional text, no explanations, no code formatting.

Example format:
{{
  "user_impact": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
  "pros": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
  "cons": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"]
}}
"""
    try:
        logger.info(f"🧠 Sending analysis request to Ollama for '{headline[:40]}...'")
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False},
            timeout=120
        )
        response.raise_for_status()
        
        response_data = response.json()
        json_string = response_data.get('response', '{}').strip()
        
        logger.info(f"📨 Raw LLM response: {json_string[:200]}...")
        
        # Clean the response - remove any markdown code blocks
        json_string = re.sub(r'```json\s*', '', json_string)
        json_string = re.sub(r'```\s*', '', json_string)
        json_string = json_string.strip()
        
        # Try to extract JSON using multiple methods
        json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
        
        # Parse the JSON
        analysis_json = json.loads(json_string)
        
        # Validate the structure
        if all(key in analysis_json for key in ["user_impact", "pros", "cons"]):
            # Validate that each value is a list
            if (isinstance(analysis_json["user_impact"], list) and 
                isinstance(analysis_json["pros"], list) and 
                isinstance(analysis_json["cons"], list)):
                
                logger.info(f"✅ Successfully received valid analysis for '{headline[:40]}...'")
                return analysis_json
        
        # If we get here, the structure is invalid
        logger.warning(f"⚠️ LLM returned invalid JSON structure for '{headline[:40]}...'")
        return {
            "user_impact": ["Analysis could not be generated for this content."],
            "pros": ["Unable to provide advantages analysis."],
            "cons": ["Unable to provide risks analysis."]
        }
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing failed for '{headline[:40]}...': {e}")
        logger.error(f"📝 Problematic response: {json_string[:500]}")
        return {
            "user_impact": ["Invalid analysis format received from AI."],
            "pros": ["Analysis generation failed due to format issues."],
            "cons": ["Unable to process the generated analysis."]
        }
    except requests.exceptions.Timeout:
        logger.error(f"❌ Ollama request timed out for '{headline[:40]}...'")
        return {
            "user_impact": ["Analysis request timed out."],
            "pros": ["Unable to generate analysis due to timeout."],
            "cons": ["Analysis service unavailable."]
        }
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Cannot connect to Ollama server for '{headline[:40]}...'")
        return {
            "user_impact": ["Analysis service unavailable."],
            "pros": ["Cannot connect to analysis engine."],
            "cons": ["Ollama server is not running."]
        }
    except Exception as e:
        logger.error(f"❌ Unexpected error in analysis for '{headline[:40]}...': {e}")
        return {
            "user_impact": ["Unexpected error during analysis."],
            "pros": ["Analysis generation failed."],
            "cons": ["Technical issues prevented analysis."]
        }

# === Main Application Class ===

class AINewsMagazine:
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.content_filter = ContentFilter()
        self.extractor_agent = ExtractorAgent()
        self.deduplicator = Deduplicator()
        self.headline_generator = HeadlineGenerator()
        self.summary_generator = SummaryGenerator()
        self.github_finder = GitHubRepoFinder()
        self.design_engine = DesignEngine()
        self.config = Config()
        
    def generate_magazine(self):
        global current_processing_article, total_articles_to_process
        logger.info("🚀 Starting Advanced Agentic Magazine Pipeline...")
        
        try:
            # Step 1: Fetch raw articles
            logger.info("Step 1: Fetching raw news articles...")
            raw_articles = self.news_fetcher.fetch_all_news()
            
            if not raw_articles:
                logger.warning("No articles fetched from news sources.")
                return self._create_empty_magazine()

            # Step 2: Classify articles
            logger.info(f"Step 2: Classifying {len(raw_articles)} articles...")
            classified_articles = self.content_filter.classify_articles(raw_articles)
            
            # Step 3: Process classifications
            single_stories = []
            aggregator_count = 0
            
            for article, classification in classified_articles:
                if classification == "SINGLE_STORY":
                    single_stories.append(article)
                elif classification == "AGGREGATOR_STORY":
                    aggregator_count += 1
                    extracted = self.extractor_agent.extract_stories(article)
                    single_stories.extend(extracted)
            
            logger.info(f"Found {len(single_stories)} single stories and {aggregator_count} aggregators")

            if not single_stories:
                logger.warning("No AI/ML articles found after classification.")
                return self._create_empty_magazine()

            # Step 4: Deduplicate
            logger.info(f"Step 4: Deduplicating {len(single_stories)} stories...")
            unique_ai_articles = self.deduplicator.remove_duplicates(single_stories)

            if not unique_ai_articles:
                logger.warning("No unique AI/ML articles found after deduplication.")
                return self._create_empty_magazine()
            
            # Step 5: Process final articles
            total_articles_to_process = len(unique_ai_articles)
            logger.info(f"Step 5: Processing {total_articles_to_process} final articles...")
            
            processed_articles = []
            for i, article in enumerate(unique_ai_articles):
                current_processing_article = i + 1
                logger.info(f"Processing article {i+1}/{total_articles_to_process}: {article.get('title', '')[:50]}...")
                
                # Generate headline and summary
                headline = self.headline_generator.generate_headline(article)
                summary = self.summary_generator.generate_comprehensive_summary(article)
                
                article_data = {
                    'original_title': article.get('title', ''),
                    'headline': headline,
                    'summary': summary,
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', '#'),
                    'published_at': article.get('publishedAt', ''),
                    'urlToImage': article.get('urlToImage', ''),
                }

                # Generate analysis with comprehensive error handling
                logger.info(f"🔍 Generating AI analysis for article {i+1}...")
                try:
                    analysis = analyze_article_impacts_with_ollama(article_data)
                    
                    # Validate that we got meaningful analysis (not just error messages)
                    is_analysis_valid = True
                    
                    # Check if any of the analysis fields contain error indicators
                    error_indicators = [
                        "Analysis could not be generated",
                        "Unable to generate analysis",
                        "Invalid analysis format",
                        "Analysis generation failed",
                        "Analysis service unavailable",
                        "Cannot connect to analysis engine",
                        "Ollama server is not running",
                        "Analysis request timed out",
                        "Unexpected error during analysis",
                        "No analysis available",
                        "Content analysis unavailable",
                        "Unable to provide"
                    ]
                    
                    # Check all analysis fields for error messages
                    for key in ["user_impact", "pros", "cons"]:
                        if key in analysis and isinstance(analysis[key], list) and analysis[key]:
                            first_item = analysis[key][0]
                            if any(indicator in first_item for indicator in error_indicators):
                                is_analysis_valid = False
                                break
                    
                    if not is_analysis_valid:
                        logger.warning(f"⚠️ Analysis failed for article {i+1}, using fallback analysis")
                        analysis = {
                            "user_impact": ["AI analysis unavailable for this article."],
                            "pros": ["No advantages analysis generated."],
                            "cons": ["No risks analysis available."]
                        }
                    else:
                        logger.info(f"✅ Analysis successfully generated for article {i+1}")
                        
                except Exception as e:
                    logger.error(f"❌ Analysis generation failed completely for article {i+1}: {e}")
                    analysis = {
                        "user_impact": ["Analysis service temporarily unavailable."],
                        "pros": ["Unable to generate analysis at this time."],
                        "cons": ["Please try again later or check AI service status."]
                    }

                article_data['analysis'] = analysis
                processed_articles.append(article_data)
                
                # Small delay to avoid overwhelming the LLM
                if i < len(unique_ai_articles) - 1:
                    time.sleep(2)
            
            # Step 6: Find GitHub repos (placeholder - implement if needed)

            # With this:
            logger.info("Finding related GitHub repositories...")
            github_repos = []
            try:
                if processed_articles:
                    # Use the first few articles to find related repos
                    for article in processed_articles[:3]:  # Limit to 3 articles to avoid rate limits
                        try:
                            repos = self.github_finder.find_related_repos(article)
                            github_repos.extend(repos)
                            time.sleep(1)  # Rate limiting
                        except Exception as e:
                            logger.warning(f"GitHub search failed for article: {e}")
                            continue
                    
                    # Remove duplicates
                    unique_repos = []
                    seen_repos = set()
                    for repo in github_repos:
                        if repo['full_name'] not in seen_repos:
                            seen_repos.add(repo['full_name'])
                            unique_repos.append(repo)
                    
                    github_repos = unique_repos[:8]  # Limit to 8 repos
                    logger.info(f"Found {len(github_repos)} unique GitHub repositories")
                    
            except Exception as e:
                logger.error(f"Error in GitHub repo finding: {e}")
                github_repos = self._get_fallback_repos()
            
            # Step 7: Create magazine layout
            logger.info("Creating magazine layout...")
            magazine_data = self.design_engine.create_magazine_layout(processed_articles, github_repos)
            
            # Log analysis statistics
            successful_analyses = sum(1 for article in processed_articles 
                                    if "AI analysis unavailable" not in article.get('analysis', {}).get('user_impact', [''])[0])
            logger.info(f"📊 Analysis Statistics: {successful_analyses}/{len(processed_articles)} articles successfully analyzed")
            
            logger.info("🎉 Magazine generation completed successfully!")
            return magazine_data
            
        except Exception as e:
            logger.error(f"❌ Error in magazine generation: {e}", exc_info=True)
            return self._create_error_magazine(str(e))

    def _create_empty_magazine(self):
        """Creates the data structure for an empty magazine."""
        return {
            'metadata': {
                'title': "AI/ML Daily Digest",
                'curator': "Nagendra Ojha", 
                'date': datetime.now().strftime("%Y-%m-%d"),
                'total_articles': 0,
                'total_repos': 0,
                'empty': True
            },
            'cover_page': {
                'main_title': "AI/ML Daily Digest",
                'subtitle': "No AI/ML News Today",
                'featured_quote': "The only constant is change.",
                'today_highlight': "No significant AI developments were found in the last 48 hours."
            },
            'articles': [],
            'github_repos': [],
            'contact_section': {
                "name": "NAGENDRA OJHA",
                "title": "AI Research Engineer & Innovation Catalyst",
                "email": "ojhanagendra04@gmail.com",
                "linkedin": "https://www.linkedin.com/in/nagendra-ojha-2k25/",
                "github": "https://github.com/nagendraojha",
                "message": "Bridging AI Research with Real-World Impact",
                "call_to_action": "CONNECT FOR AI INNOVATION"
            },
            'footer': {
                "text": "© 2025 AI/ML DAILY DIGEST",
                "curator_note": "Curated with 🤖 by Nagendra Ojha",
                "disclaimer": "AI-generated intelligence. Verify critical data from primary sources.",
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "version": "QUANTUM EDITION v2.0"
            }
        }
    
    def _create_error_magazine(self, error_msg):
        """Creates the data structure for an error state."""
        return {
            'metadata': {
                'title': "Magazine Generation Error",
                'curator': "Nagendra Ojha",
                'date': datetime.now().strftime("%Y-%m-%d"),
                'error': True,
                'error_message': str(error_msg)
            },
            'cover_page': {
                'main_title': "Generation Error",
                'subtitle': "Could not create the magazine",
                'featured_quote': "Errors are portals of discovery.",
                'today_highlight': f"An error occurred: {str(error_msg)[:100]}..."
            },
            'articles': [],
            'github_repos': [],
            'contact_section': {
                "name": "NAGENDRA OJHA",
                "title": "AI Research Engineer & Innovation Catalyst",
                "email": "ojhanagendra04@gmail.com",
                "linkedin": "https://www.linkedin.com/in/nagendra-ojha-2k25/",
                "github": "https://github.com/nagendraojha",
                "message": "Bridging AI Research with Real-World Impact",
                "call_to_action": "CONNECT FOR AI INNOVATION"
            },
            'footer': {
                "text": "© 2025 AI/ML DAILY DIGEST",
                "curator_note": "Curated with 🤖 by Nagendra Ojha",
                "disclaimer": "AI-generated intelligence. Verify critical data from primary sources.",
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "version": "QUANTUM EDITION v2.0"
            }
        }

# === PDF GENERATION ===

def create_futuristic_pdf(magazine):
    """Creates a PDF with proper text wrapping for analysis content."""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            topMargin=0.5*inch, 
            bottomMargin=0.5*inch, 
            leftMargin=0.5*inch, 
            rightMargin=0.5*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Color scheme
        DEEP_BLUE = HexColor('#0D47A1')
        LIGHT_BLUE = HexColor('#42A5F5')
        ACCENT_GREEN = HexColor('#00C853')
        ACCENT_RED = HexColor('#D50000')
        DARK_GRAY = HexColor('#212121')
        MID_GRAY = HexColor('#757575')
        LIGHT_GRAY = HexColor('#F5F5F5')
        
        # Custom styles with better text wrapping
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=DEEP_BLUE,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            wordWrap='LTR'
        )
        
        article_title_style = ParagraphStyle(
            'ArticleTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=DEEP_BLUE,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            wordWrap='LTR'
        )
        
        source_style = ParagraphStyle(
            'Source',
            fontSize=9,
            fontName='Helvetica-Oblique',
            textColor=MID_GRAY,
            spaceAfter=12,
            wordWrap='LTR'
        )
        
        summary_style = ParagraphStyle(
            'Summary',
            fontSize=10,
            fontName='Helvetica',
            textColor=DARK_GRAY,
            alignment=TA_JUSTIFY,
            leading=14,
            spaceAfter=12,
            wordWrap='LTR'
        )
        
        # Improved styles for analysis content
        analysis_item_style = ParagraphStyle(
            'AnalysisItem',
            fontSize=8,
            fontName='Helvetica',
            textColor=DARK_GRAY,
            leading=10,
            spaceAfter=4,
            leftIndent=0,
            rightIndent=0,
            wordWrap='LTR',
            splitLongWords=True,
            alignment=TA_LEFT
        )
        
        section_header_style = ParagraphStyle(
            'SectionHeader',
            fontSize=10,
            fontName='Helvetica-Bold',
            textColor=white,
            alignment=TA_CENTER,
            wordWrap='LTR'
        )
        
        story = []
        
        # Cover page
        cover_page = magazine.get('cover_page', {})
        story.append(Paragraph("AI/ML DAILY DIGEST", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(cover_page.get('subtitle', 'Your Daily AI News'), article_title_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Date: {magazine.get('metadata', {}).get('date', '')}", source_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Featured quote
        featured_quote = cover_page.get('featured_quote', '')
        if featured_quote:
            story.append(Paragraph(f'"{featured_quote}"', summary_style))
            story.append(Spacer(1, 0.3*inch))
        
        story.append(PageBreak())
        
        # Articles with improved analysis formatting
        articles = magazine.get('articles', [])
        for i, article in enumerate(articles):
            if i >= 500:  # Limit to 8 articles for PDF
                break
                
            # Article header
            story.append(Paragraph(f"Article {i+1}", title_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Headline and source
            headline = article.get('headline', 'No headline')
            # Truncate very long headlines
            if len(headline) > 300:
                headline = headline[:299] + "..."
            story.append(Paragraph(headline, article_title_style))
            story.append(Paragraph(f"Source: {article.get('source', 'Unknown')}", source_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Summary with truncation for very long content
            summary = article.get('summary', 'No summary available.')
            if len(summary) > 5000:
                summary = summary[:4000] + "..."
            story.append(Paragraph(summary, summary_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Analysis sections with better formatting
            analysis = article.get('analysis', {})
            
            # User Impact - with text truncation
            user_impact = analysis.get('user_impact', ["Analysis not available."])
            if user_impact:
                impact_header = Paragraph("User Impact Analysis", section_header_style)
                impact_data = [[impact_header]]
                
                for point in user_impact[:5]:  # Limit to 3 points
                    # Truncate long points
                    if len(point) > 3000:
                        point = point[:2900] + "..."
                    impact_data.append([Paragraph(f"• {point}", analysis_item_style)])
                
                impact_table = Table(impact_data, colWidths=doc.width)
                impact_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BLUE),
                    ('TEXTCOLOR', (0, 0), (-1, 0), white),
                    ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOX', (0, 0), (-1, -1), 1, DARK_GRAY),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(impact_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Pros and Cons side by side with better formatting
            pros = analysis.get('pros', [])
            cons = analysis.get('cons', [])
            
            if pros or cons:
                # Create data for the table
                table_data = [["Advantages (Pros)", "Challenges (Cons)"]]
                
                # Add items - take the maximum length
                max_len = max(len(pros), len(cons))
                for j in range(min(max_len, 5)):  # Limit to 3 items each
                    pro_item = pros[j] if j < len(pros) else ""
                    con_item = cons[j] if j < len(cons) else ""
                    
                    # Truncate long items
                    if len(pro_item) > 3000:
                        pro_item = pro_item[:2900] + "..."
                    if len(con_item) > 3000:
                        con_item = con_item[:2900] + "..."
                        
                    table_data.append([
                        Paragraph(f"• {pro_item}", analysis_item_style) if pro_item else "",
                        Paragraph(f"• {con_item}", analysis_item_style) if con_item else ""
                    ])
                
                pros_cons_table = Table(
                    table_data, 
                    colWidths=[doc.width/2 - 0.2*inch, doc.width/2 - 0.2*inch]
                )
                pros_cons_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), ACCENT_GREEN),
                    ('BACKGROUND', (1, 0), (1, 0), ACCENT_RED),
                    ('TEXTCOLOR', (0, 0), (-1, 0), white),
                    ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOX', (0, 0), (-1, -1), 1, DARK_GRAY),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(pros_cons_table)
                story.append(Spacer(1, 0.3*inch))
            
            if i < len(articles) - 1:
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        logger.info("✅ PDF generated successfully")
        return pdf_data
        
    except Exception as e:
        logger.error(f"❌ Error creating PDF: {e}")
        raise

def _get_fallback_repos(self):
    """Provide fallback GitHub repos when search fails"""
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
            "name": "transformers",
            "full_name": "huggingface/transformers",
            "description": "State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow",
            "url": "https://github.com/huggingface/transformers", 
            "stars": 89000,
            "forks": 19000,
            "language": "Python",
            "topics": ["nlp", "pytorch", "tensorflow", "jax"],
        },
        {
            "name": "pytorch",
            "full_name": "pytorch/pytorch",
            "description": "Tensors and Dynamic neural networks in Python with strong GPU acceleration",
            "url": "https://github.com/pytorch/pytorch",
            "stars": 68000,
            "forks": 18500,
            "language": "Python", 
            "topics": ["deep-learning", "python", "gpu"],
        }
    ]

# === Global variables and Flask Routes ===
magazine_generator = AINewsMagazine()
current_magazine = None
is_generating = False
generation_start_time = 0
current_processing_article = 0
total_articles_to_process = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate_magazine_route():
    global current_magazine, is_generating, generation_start_time
    if is_generating:
        return jsonify({"status": "already_generating"})
    
    is_generating = True
    generation_start_time = time.time()
    try:
        current_magazine = magazine_generator.generate_magazine()
        return jsonify(current_magazine)
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)})
    finally:
        is_generating = False

@app.route('/magazine')
def view_magazine():
    global current_magazine, is_generating
    if is_generating:
        elapsed = time.time() - generation_start_time
        progress = f"{current_processing_article}/{total_articles_to_process}" if total_articles_to_process > 0 else "starting..."
        return f"""
        <html>
            <head><title>Generating Magazine...</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1>🔄 Generating AI/ML Magazine...</h1>
                <p>This may take 10-30 minutes. Please wait.</p>
                <p><strong>Progress:</strong> Processing article {progress}</p>
                <p><strong>Elapsed time:</strong> {int(elapsed)} seconds</p>
                <p><em>Please don't refresh or open multiple tabs</em></p>
                <p><a href="/magazine">Check progress</a> | <a href="/">Home</a></p>
            </body>
        </html>
        """
    
    if current_magazine is None:
        return "No magazine generated yet. Please visit the homepage to generate one."
    
    return render_template('magazine.html', magazine=current_magazine)

@app.route('/status')
def generation_status():
    global is_generating, generation_start_time, current_processing_article, total_articles_to_process
    elapsed = time.time() - generation_start_time if is_generating else 0
    return jsonify({
        "is_generating": is_generating,
        "current_article": current_processing_article,
        "total_articles": total_articles_to_process,
        "elapsed_seconds": int(elapsed),
        "has_magazine": current_magazine is not None
    })

@app.route('/download-pdf')
def download_pdf_route():
    global current_magazine
    if current_magazine is None:
        logger.info("Download requested but no magazine found. Generating a new one.")
        try:
            current_magazine = magazine_generator.generate_magazine()
        except Exception as e:
            logger.error(f"Failed to generate magazine for download: {e}")
            return "Error generating magazine for download.", 500
    
    try:
        pdf_content = create_futuristic_pdf(current_magazine)
        return pdf_content, 200, {
            'Content-Type': 'application/pdf', 
            'Content-Disposition': 'attachment; filename=AI-ML-Magazine.pdf'
        }
    except Exception as e:
        logger.error(f"Error creating PDF: {e}", exc_info=True)
        return "Error creating PDF.", 500

if __name__ == '__main__':
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=5000)