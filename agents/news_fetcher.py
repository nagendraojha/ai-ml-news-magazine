import logging
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone
import time
import requests
import feedparser
from bs4 import BeautifulSoup
import concurrent.futures
from utils.config import Config
from utils.ollama_client import OllamaClient

class NewsFetcher:
    def __init__(self):
        self.config = Config
        self.ollama = OllamaClient()
        self.logger = logging.getLogger(__name__)
        # Use a 48-hour window with explicit timezone
        self.since_date = datetime.now(timezone.utc) - timedelta(hours=48)
        self.timeout_seconds = 10

    def _is_recent(self, published_struct) -> bool:
        """Improved date checking with multiple fallback strategies."""
        if not published_struct:
            return False
        
        try:
            # Method 1: Try feedparser's built-in parsing first
            if hasattr(published_struct, 'parsed'):
                published_date = published_struct.parsed.replace(tzinfo=timezone.utc)
                return published_date >= self.since_date
            
            # Method 2: Handle time.struct_time with proper UTC conversion
            if isinstance(published_struct, time.struct_time):
                # Convert struct_time to datetime assuming UTC
                published_date = datetime(
                    published_struct.tm_year, 
                    published_struct.tm_mon, 
                    published_struct.tm_mday,
                    published_struct.tm_hour, 
                    published_struct.tm_min, 
                    published_struct.tm_sec,
                    tzinfo=timezone.utc
                )
                return published_date >= self.since_date
            
            # Method 3: Handle string dates (fallback)
            if isinstance(published_struct, str):
                # Try common date formats
                for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z']:
                    try:
                        published_date = datetime.strptime(published_struct, fmt)
                        if published_date.tzinfo is None:
                            published_date = published_date.replace(tzinfo=timezone.utc)
                        return published_date >= self.since_date
                    except ValueError:
                        continue
            
            return False
            
        except (TypeError, ValueError, AttributeError) as e:
            self.logger.warning(f"Date parsing failed for {published_struct}: {e}")
            # Conservative approach: if we can't parse date, assume it's recent
            # to avoid missing potentially relevant news
            return True  # Or return False if you want to be strict

    def _fetch_feed_with_validation(self, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch feed with enhanced date validation and logging."""
        try:
            self.logger.info(f"Fetching from: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            recent_articles = []
            for entry in feed.entries:
                # Log the date for debugging
                published = getattr(entry, 'published_parsed', None) or getattr(entry, 'updated_parsed', None)
                self.logger.debug(f"Article date: {published}, Title: {getattr(entry, 'title', 'No title')}")
                
                if self._is_recent(published):
                    recent_articles.append({
                        'title': getattr(entry, 'title', 'No title'),
                        'link': getattr(entry, 'link', ''),
                        'published': published,
                        'summary': getattr(entry, 'summary', ''),
                        'source': feed_url
                    })
                else:
                    self.logger.debug(f"Filtered out old article: {getattr(entry, 'title', 'No title')}")
            
            self.logger.info(f"Found {len(recent_articles)} recent articles from {feed_url}")
            return recent_articles
            
        except Exception as e:
            self.logger.error(f"Error fetching feed {feed_url}: {e}")
            return []

    def add_debug_logging(self):
        """Add this method to debug your current feeds."""
        for feed_url in self.config.RSS_FEEDS:
            print(f"\n=== Checking: {feed_url} ===")
            feed = feedparser.parse(feed_url)
            for i, entry in enumerate(feed.entries[:3]):  # Check first 3 entries
                published = getattr(entry, 'published_parsed', None)
                title = getattr(entry, 'title', 'No title')
                is_recent = self._is_recent(published)
                print(f"{i+1}. {title[:50]}... | Recent: {is_recent} | Date: {published}")

    def _get_global_news_sources(self) -> Dict[str, str]:
        """Returns top English news sources from every country."""
        return {
            # MAJOR GLOBAL NEWS NETWORKS
            "Reuters": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
            "Associated Press": "https://apnews.com/apf-topnews",
            "CNN": "https://rss.cnn.com/rss/edition.rss",
            "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
            "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
            "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
            "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
            "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "BBC Science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "BBC Health": "https://feeds.bbci.co.uk/news/health/rss.xml",
            "BBC Entertainment": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "BBC Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
            "BBC Education": "https://feeds.bbci.co.uk/news/education/rss.xml",
            "BBC Asia": "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
            "BBC Europe": "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
            "BBC Middle East": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
            "BBC Africa": "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
            "BBC US & Canada": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
            "BBC Latin America": "https://feeds.bbci.co.uk/news/world/latin_america/rss.xml",
            "BBC UK": "https://feeds.bbci.co.uk/news/england/rss.xml",
            "BBC Scotland": "https://feeds.bbci.co.uk/news/scotland/rss.xml",
            "BBC Wales": "https://feeds.bbci.co.uk/news/wales/rss.xml",
            "BBC Northern Ireland": "https://feeds.bbci.co.uk/news/northern_ireland/rss.xml",

            # US NATIONAL NEWS
            "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "NYT Business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
            "NYT Technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
            "NYT Science": "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
            "NYT Health": "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
            "NYT Politics": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
            "NYT Sports": "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
            "NYT Arts": "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
            "NYT World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "NYT US": "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
            "Washington Post": "https://feeds.washingtonpost.com/rss/world",
            "Wall Street Journal": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
            "WSJ US Business": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
            "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
            "WSJ Technology": "https://feeds.a.dj.com/rss/RSSTechnology.xml",
            "NBC News": "https://feeds.nbcnews.com/nbcnews/public/news",
            "NBC Politics": "https://feeds.nbcnews.com/nbcnews/public/politics",
            "NBC World": "https://feeds.nbcnews.com/nbcnews/public/world",
            "NBC Business": "https://feeds.nbcnews.com/nbcnews/public/business",
            "ABC News": "https://abcnews.go.com/abcnews/topstories",
            "ABC Politics": "https://abcnews.go.com/abcnews/politicsheadlines",
            "ABC World": "https://abcnews.go.com/abcnews/internationalheadlines",
            "CBS News": "https://cbsnews.com/rss/news/",
            "CBS World": "https://cbsnews.com/rss/world/",
            "CBS Politics": "https://cbsnews.com/rss/politics/",
            "Fox News": "https://moxie.foxnews.com/google-publisher/latest.xml",
            "Fox Politics": "https://moxie.foxnews.com/google-publisher/politics.xml",
            "Fox World": "https://moxie.foxnews.com/google-publisher/world.xml",
            "USA Today": "https://rssfeeds.usatoday.com/usatoday-news-top-stories",
            "USA Today World": "https://rssfeeds.usatoday.com/UsatodaycomWorld-TopStories",
            "USA Today Politics": "https://rssfeeds.usatoday.com/UsatodaycomWashington-TopStories",
            "PBS News": "https://www.pbs.org/newshour/feed/",
            "NPR News": "https://feeds.npr.org/1001/rss.xml",
            "NPR World": "https://feeds.npr.org/1004/rss.xml",
            "NPR Politics": "https://feeds.npr.org/1014/rss.xml",
            "NPR Business": "https://feeds.npr.org/1006/rss.xml",
            "NPR Technology": "https://feeds.npr.org/1019/rss.xml",
            "NPR Science": "https://feeds.npr.org/1007/rss.xml",
            "Bloomberg": "https://feeds.bloomberg.com/bloomberg/news.rss",
            "Bloomberg Markets": "https://feeds.bloomberg.com/bloomberg/markets/news.rss",
            "Bloomberg Technology": "https://feeds.bloomberg.com/bloomberg/technology/news.rss",
            "Bloomberg Politics": "https://feeds.bloomberg.com/bloomberg/politics/news.rss",
            "Time": "https://time.com/feed/",
            "Newsweek": "https://www.newsweek.com/rss",
            "US News": "https://www.usnews.com/rss/news",
            "The Hill": "https://thehill.com/news/feed/",
            "Politico": "https://www.politico.com/rss/politicopicks.xml",
            "Politico Congress": "https://www.politico.com/rss/congress.xml",
            "Roll Call": "https://rollcall.com/feed/",
            "The Atlantic": "https://www.theatlantic.com/feed/all/",
            "New Yorker": "https://www.newyorker.com/feed/news",
            "HuffPost": "https://www.huffpost.com/section/front-page/feed",
            "HuffPost Politics": "https://www.huffpost.com/section/politics/feed",
            "HuffPost World": "https://www.huffpost.com/section/world-news/feed",
            "Vox": "https://www.vox.com/rss/index.xml",
            "BuzzFeed News": "https://www.buzzfeednews.com/article/rss.xml",
            "ProPublica": "https://www.propublica.org/feeds/propublica/main",
            "Mother Jones": "https://www.motherjones.com/feed/",
            "The Intercept": "https://theintercept.com/feed/?lang=en",
            "Democracy Now": "https://www.democracynow.org/democracynow.rss",
            "The Nation": "https://www.thenation.com/feed/",
            "National Review": "https://www.nationalreview.com/feed/",
            "The American Conservative": "https://www.theamericanconservative.com/feed/",
            "Reason": "https://reason.com/feed/",
            "The Federalist": "https://thefederalist.com/feed/",
            "Breitbart": "https://www.breitbart.com/feed/",
            "The Daily Caller": "https://dailycaller.com/feed/",
            "Washington Times": "https://www.washingtontimes.com/rss/headlines/news/",
            "Washington Examiner": "https://www.washingtonexaminer.com/rss/news",
            "New York Post": "https://nypost.com/news/feed/",
            "Chicago Tribune": "https://www.chicagotribune.com/arcio/rss/",
            "LA Times": "https://www.latimes.com/rss/",
            "Boston Globe": "https://www.bostonglobe.com/rss/",
            "Miami Herald": "https://www.miamiherald.com/latest-news/rss/",
            "Houston Chronicle": "https://www.houstonchronicle.com/rss/",
            "Dallas Morning News": "https://www.dallasnews.com/arcio/rss/",
            "Denver Post": "https://www.denverpost.com/feed/",
            "Seattle Times": "https://www.seattletimes.com/rss/",
            "San Francisco Chronicle": "https://www.sfchronicle.com/rss/",
            "Arizona Republic": "https://www.azcentral.com/rss/",
            "Atlanta Journal-Constitution": "https://www.ajc.com/rss/",

            # UK NEWS SOURCES
            "The Guardian": "https://www.theguardian.com/international/rss",
            "Guardian UK": "https://www.theguardian.com/uk/rss",
            "Guardian World": "https://www.theguardian.com/world/rss",
            "Guardian Politics": "https://www.theguardian.com/politics/rss",
            "Guardian Business": "https://www.theguardian.com/uk/business/rss",
            "Guardian Technology": "https://www.theguardian.com/uk/technology/rss",
            "Guardian Science": "https://www.theguardian.com/science/rss",
            "Guardian Environment": "https://www.theguardian.com/uk/environment/rss",
            "The Times": "https://www.thetimes.co.uk/rss",
            "The Telegraph": "https://www.telegraph.co.uk/rss.xml",
            "The Independent": "https://www.independent.co.uk/news/rss",
            "Daily Mail": "https://www.dailymail.co.uk/news/index.rss",
            "Daily Express": "https://www.express.co.uk/posts/rss/1/feed",
            "The Sun": "https://www.thesun.co.uk/news/feed/",
            "Mirror": "https://www.mirror.co.uk/news/?service=rss",
            "Evening Standard": "https://www.standard.co.uk/news/rss",
            "Financial Times": "https://www.ft.com/rss/home",
            "FT World": "https://www.ft.com/world?format=rss",
            "FT UK": "https://www.ft.com/uk?format=rss",
            "FT US": "https://www.ft.com/us?format=rss",
            "FT Europe": "https://www.ft.com/europe?format=rss",
            "FT Asia": "https://www.ft.com/asia-pacific?format=rss",
            "Sky News": "https://feeds.skynews.com/feeds/rss/world.xml",
            "Sky UK": "https://feeds.skynews.com/feeds/rss/uk.xml",
            "Sky Politics": "https://feeds.skynews.com/feeds/rss/politics.xml",
            "Sky Business": "https://feeds.skynews.com/feeds/rss/business.xml",
            "Sky Technology": "https://feeds.skynews.com/feeds/rss/technology.xml",
            "ITV News": "https://www.itv.com/news/feed",
            "Channel 4 News": "https://www.channel4.com/news/feed",
            "The Economist": "https://www.economist.com/rss/all_rss_content.xml",
            "The Spectator": "https://www.spectator.co.uk/feed/rss",

            # CANADIAN NEWS
            "CBC News": "https://rss.cbc.ca/lineup/topstories.xml",
            "CBC World": "https://rss.cbc.ca/lineup/world.xml",
            "CBC Politics": "https://rss.cbc.ca/lineup/politics.xml",
            "CBC Business": "https://rss.cbc.ca/lineup/business.xml",
            "CTV News": "https://www.ctvnews.ca/rss/ctv-news-cp-24",
            "Global News": "https://globalnews.ca/feed/",
            "Toronto Star": "https://www.thestar.com/rss.xml",
            "National Post": "https://nationalpost.com/feed/",
            "The Globe and Mail": "https://www.theglobeandmail.com/rss/",
            "Vancouver Sun": "https://vancouversun.com/feed/",
            "Montreal Gazette": "https://montrealgazette.com/feed/",
            "Calgary Herald": "https://calgaryherald.com/feed/",
            "Ottawa Citizen": "https://ottawacitizen.com/feed/",

            # AUSTRALIAN & NEW ZEALAND
            "ABC Australia": "https://www.abc.net.au/news/feed/51120/rss.xml",
            "ABC World": "https://www.abc.net.au/news/feed/52278/rss.xml",
            "ABC Politics": "https://www.abc.net.au/news/feed/52292/rss.xml",
            "ABC Business": "https://www.abc.net.au/news/feed/51892/rss.xml",
            "Sydney Morning Herald": "https://www.smh.com.au/rss/feed.xml",
            "The Age": "https://www.theage.com.au/rss/feed.xml",
            "The Australian": "https://www.theaustralian.com.au/feed/",
            "News.com.au": "https://www.news.com.au/feed/",
            "NZ Herald": "https://www.nzherald.co.nz/rss/",
            "Stuff NZ": "https://www.stuff.co.nz/rss/",
            "Radio New Zealand": "https://www.rnz.co.nz/rss/news.xml",

            # EUROPEAN ENGLISH SOURCES
            "France 24": "https://www.france24.com/en/rss",
            "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
            "DW Europe": "https://rss.dw.com/rdf/rss-en-eu",
            "DW World": "https://rss.dw.com/rdf/rss-en-world",
            "DW Business": "https://rss.dw.com/rdf/rss-en-business",
            "DW Science": "https://rss.dw.com/rdf/rss-en-sci",
            "Euronews": "https://www.euronews.com/rss",
            "EU Observer": "https://euobserver.com/rss",
            "Politico Europe": "https://www.politico.eu/feed/",
            "The Local Europe": "https://www.thelocal.com/feed/",
            "Irish Times": "https://www.irishtimes.com/rss/",
            "Irish Independent": "https://www.independent.ie/rss/",
            "RTÉ News": "https://www.rte.ie/rss/news/",
            "The Journal.ie": "https://www.thejournal.ie/feed/",
            "Dutch News": "https://www.dutchnews.nl/feed/",
            "The Brussels Times": "https://www.brusselstimes.com/feed/",
            "Swiss Info": "https://www.swissinfo.ch/eng/rss/",
            "Anadolu Agency": "https://www.aa.com.tr/en/rss/default?cat=general",

            # ASIAN ENGLISH SOURCES
            "South China Morning Post": "https://www.scmp.com/rss/91/feed",
            "SCMP China": "https://www.scmp.com/rss/2/feed",
            "SCMP Asia": "https://www.scmp.com/rss/4/feed",
            "SCMP World": "https://www.scmp.com/rss/3/feed",
            "SCMP Business": "https://www.scmp.com/rss/18/feed",
            "SCMP Technology": "https://www.scmp.com/rss/36/feed",
            "Japan Times": "https://www.japantimes.co.jp/feed/",
            "The Straits Times": "https://www.straitstimes.com/news/singapore/rss.xml",
            "Straits Times Asia": "https://www.straitstimes.com/news/asia/rss.xml",
            "Straits Times World": "https://www.straitstimes.com/news/world/rss.xml",
            "Straits Times Business": "https://www.straitstimes.com/news/business/rss.xml",
            "Channel News Asia": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3764/rss/",
            "CNA Asia": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3216/rss/",
            "CNA World": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3218/rss/",
            "CNA Business": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3214/rss/",
            "Manila Bulletin": "https://mb.com.ph/feed/",
            "Philippine Star": "https://www.philstar.com/rss/headlines",
            "Bangkok Post": "https://www.bangkokpost.com/rss/data/feed.xml",
            "The Nation Thailand": "https://www.nationthailand.com/rss/",
            "Jakarta Post": "https://www.thejakartapost.com/rss/",
            "Korea Herald": "https://www.koreaherald.com/rss/",
            "Korea Times": "https://www.koreatimes.co.kr/rss/",
            "Taiwan News": "https://www.taiwannews.com.tw/rss/",
            "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
            "TOI World": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
            "TOI India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
            "TOI Business": "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
            "TOI Technology": "https://timesofindia.indiatimes.com/rssfeeds/5880659.cms",
            "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/",
            "The Hindu": "https://www.thehindu.com/feeder/default.rss",
            "Indian Express": "https://indianexpress.com/feed/",
            "Mint": "https://www.livemint.com/rss/news",
            "Economic Times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
            "The Print": "https://theprint.in/feed/",
            "Gulf News": "https://gulfnews.com/feed/",
            "Khaleej Times": "https://www.khaleejtimes.com/rss/",
            "The National UAE": "https://www.thenationalnews.com/feed/",
            "Arab News": "https://www.arabnews.com/rss.xml",
            "Jerusalem Post": "https://www.jpost.com/Rss/RssFeedsHeadlines.aspx",
            "Haaretz": "https://www.haaretz.com/rss/",
            "Times of Israel": "https://www.timesofisrael.com/feed/",

            # ... (keep all your other sources - truncated for brevity)
            # TECH NEWS
            "TechCrunch": "https://techcrunch.com/feed/",
            "The Verge": "https://www.theverge.com/rss/index.xml",
            "Wired": "https://www.wired.com/feed/rss",
            "Ars Technica": "https://arstechnica.com/feed/",
             # MAJOR GLOBAL NEWS NETWORKS
            "Reuters": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
            "Associated Press": "https://apnews.com/apf-topnews",
            "CNN": "https://rss.cnn.com/rss/edition.rss",
            "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
            "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
            "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
            "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
            "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "BBC Science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "BBC Health": "https://feeds.bbci.co.uk/news/health/rss.xml",
            "BBC Entertainment": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "BBC Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
            "BBC Education": "https://feeds.bbci.co.uk/news/education/rss.xml",
            "BBC Asia": "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
            "BBC Europe": "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
            "BBC Middle East": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
            "BBC Africa": "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
            "BBC US & Canada": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
            "BBC Latin America": "https://feeds.bbci.co.uk/news/world/latin_america/rss.xml",
            "BBC UK": "https://feeds.bbci.co.uk/news/england/rss.xml",
            "BBC Scotland": "https://feeds.bbci.co.uk/news/scotland/rss.xml",
            "BBC Wales": "https://feeds.bbci.co.uk/news/wales/rss.xml",
            "BBC Northern Ireland": "https://feeds.bbci.co.uk/news/northern_ireland/rss.xml",

            # US NATIONAL NEWS
            "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "NYT Business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
            "NYT Technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
            "NYT Science": "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
            "NYT Health": "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
            "NYT Politics": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
            "NYT Sports": "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
            "NYT Arts": "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
            "NYT World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "NYT US": "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
            "Washington Post": "https://feeds.washingtonpost.com/rss/world",
            "Wall Street Journal": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
            "WSJ US Business": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
            "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
            "WSJ Technology": "https://feeds.a.dj.com/rss/RSSTechnology.xml",
            "NBC News": "https://feeds.nbcnews.com/nbcnews/public/news",
            "NBC Politics": "https://feeds.nbcnews.com/nbcnews/public/politics",
            "NBC World": "https://feeds.nbcnews.com/nbcnews/public/world",
            "NBC Business": "https://feeds.nbcnews.com/nbcnews/public/business",
            "ABC News": "https://abcnews.go.com/abcnews/topstories",
            "ABC Politics": "https://abcnews.go.com/abcnews/politicsheadlines",
            "ABC World": "https://abcnews.go.com/abcnews/internationalheadlines",
            "CBS News": "https://cbsnews.com/rss/news/",
            "CBS World": "https://cbsnews.com/rss/world/",
            "CBS Politics": "https://cbsnews.com/rss/politics/",
            "Fox News": "https://moxie.foxnews.com/google-publisher/latest.xml",
            "Fox Politics": "https://moxie.foxnews.com/google-publisher/politics.xml",
            "Fox World": "https://moxie.foxnews.com/google-publisher/world.xml",
            "USA Today": "https://rssfeeds.usatoday.com/usatoday-news-top-stories",
            "USA Today World": "https://rssfeeds.usatoday.com/UsatodaycomWorld-TopStories",
            "USA Today Politics": "https://rssfeeds.usatoday.com/UsatodaycomWashington-TopStories",
            "PBS News": "https://www.pbs.org/newshour/feed/",
            "NPR News": "https://feeds.npr.org/1001/rss.xml",
            "NPR World": "https://feeds.npr.org/1004/rss.xml",
            "NPR Politics": "https://feeds.npr.org/1014/rss.xml",
            "NPR Business": "https://feeds.npr.org/1006/rss.xml",
            "NPR Technology": "https://feeds.npr.org/1019/rss.xml",
            "NPR Science": "https://feeds.npr.org/1007/rss.xml",
            "Bloomberg": "https://feeds.bloomberg.com/bloomberg/news.rss",
            "Bloomberg Markets": "https://feeds.bloomberg.com/bloomberg/markets/news.rss",
            "Bloomberg Technology": "https://feeds.bloomberg.com/bloomberg/technology/news.rss",
            "Bloomberg Politics": "https://feeds.bloomberg.com/bloomberg/politics/news.rss",
            "Time": "https://time.com/feed/",
            "Newsweek": "https://www.newsweek.com/rss",
            "US News": "https://www.usnews.com/rss/news",
            "The Hill": "https://thehill.com/news/feed/",
            "Politico": "https://www.politico.com/rss/politicopicks.xml",
            "Politico Congress": "https://www.politico.com/rss/congress.xml",
            "Roll Call": "https://rollcall.com/feed/",
            "The Atlantic": "https://www.theatlantic.com/feed/all/",
            "New Yorker": "https://www.newyorker.com/feed/news",
            "HuffPost": "https://www.huffpost.com/section/front-page/feed",
            "HuffPost Politics": "https://www.huffpost.com/section/politics/feed",
            "HuffPost World": "https://www.huffpost.com/section/world-news/feed",
            "Vox": "https://www.vox.com/rss/index.xml",
            "BuzzFeed News": "https://www.buzzfeednews.com/article/rss.xml",
            "ProPublica": "https://www.propublica.org/feeds/propublica/main",
            "Mother Jones": "https://www.motherjones.com/feed/",
            "The Intercept": "https://theintercept.com/feed/?lang=en",
            "Democracy Now": "https://www.democracynow.org/democracynow.rss",
            "The Nation": "https://www.thenation.com/feed/",
            "National Review": "https://www.nationalreview.com/feed/",
            "The American Conservative": "https://www.theamericanconservative.com/feed/",
            "Reason": "https://reason.com/feed/",
            "The Federalist": "https://thefederalist.com/feed/",
            "Breitbart": "https://www.breitbart.com/feed/",
            "The Daily Caller": "https://dailycaller.com/feed/",
            "Washington Times": "https://www.washingtontimes.com/rss/headlines/news/",
            "Washington Examiner": "https://www.washingtonexaminer.com/rss/news",
            "New York Post": "https://nypost.com/news/feed/",
            "Chicago Tribune": "https://www.chicagotribune.com/arcio/rss/",
            "LA Times": "https://www.latimes.com/rss/",
            "Boston Globe": "https://www.bostonglobe.com/rss/",
            "Miami Herald": "https://www.miamiherald.com/latest-news/rss/",
            "Houston Chronicle": "https://www.houstonchronicle.com/rss/",
            "Dallas Morning News": "https://www.dallasnews.com/arcio/rss/",
            "Denver Post": "https://www.denverpost.com/feed/",
            "Seattle Times": "https://www.seattletimes.com/rss/",
            "San Francisco Chronicle": "https://www.sfchronicle.com/rss/",
            "Arizona Republic": "https://www.azcentral.com/rss/",
            "Atlanta Journal-Constitution": "https://www.ajc.com/rss/",

            # UK NEWS SOURCES
            "The Guardian": "https://www.theguardian.com/international/rss",
            "Guardian UK": "https://www.theguardian.com/uk/rss",
            "Guardian World": "https://www.theguardian.com/world/rss",
            "Guardian Politics": "https://www.theguardian.com/politics/rss",
            "Guardian Business": "https://www.theguardian.com/uk/business/rss",
            "Guardian Technology": "https://www.theguardian.com/uk/technology/rss",
            "Guardian Science": "https://www.theguardian.com/science/rss",
            "Guardian Environment": "https://www.theguardian.com/uk/environment/rss",
            "The Times": "https://www.thetimes.co.uk/rss",
            "The Telegraph": "https://www.telegraph.co.uk/rss.xml",
            "The Independent": "https://www.independent.co.uk/news/rss",
            "Daily Mail": "https://www.dailymail.co.uk/news/index.rss",
            "Daily Express": "https://www.express.co.uk/posts/rss/1/feed",
            "The Sun": "https://www.thesun.co.uk/news/feed/",
            "Mirror": "https://www.mirror.co.uk/news/?service=rss",
            "Evening Standard": "https://www.standard.co.uk/news/rss",
            "Financial Times": "https://www.ft.com/rss/home",
            "FT World": "https://www.ft.com/world?format=rss",
            "FT UK": "https://www.ft.com/uk?format=rss",
            "FT US": "https://www.ft.com/us?format=rss",
            "FT Europe": "https://www.ft.com/europe?format=rss",
            "FT Asia": "https://www.ft.com/asia-pacific?format=rss",
            "Sky News": "https://feeds.skynews.com/feeds/rss/world.xml",
            "Sky UK": "https://feeds.skynews.com/feeds/rss/uk.xml",
            "Sky Politics": "https://feeds.skynews.com/feeds/rss/politics.xml",
            "Sky Business": "https://feeds.skynews.com/feeds/rss/business.xml",
            "Sky Technology": "https://feeds.skynews.com/feeds/rss/technology.xml",
            "ITV News": "https://www.itv.com/news/feed",
            "Channel 4 News": "https://www.channel4.com/news/feed",
            "The Economist": "https://www.economist.com/rss/all_rss_content.xml",
            "The Spectator": "https://www.spectator.co.uk/feed/rss",

            # CANADIAN NEWS
            "CBC News": "https://rss.cbc.ca/lineup/topstories.xml",
            "CBC World": "https://rss.cbc.ca/lineup/world.xml",
            "CBC Politics": "https://rss.cbc.ca/lineup/politics.xml",
            "CBC Business": "https://rss.cbc.ca/lineup/business.xml",
            "CTV News": "https://www.ctvnews.ca/rss/ctv-news-cp-24",
            "Global News": "https://globalnews.ca/feed/",
            "Toronto Star": "https://www.thestar.com/rss.xml",
            "National Post": "https://nationalpost.com/feed/",
            "The Globe and Mail": "https://www.theglobeandmail.com/rss/",
            "Vancouver Sun": "https://vancouversun.com/feed/",
            "Montreal Gazette": "https://montrealgazette.com/feed/",
            "Calgary Herald": "https://calgaryherald.com/feed/",
            "Ottawa Citizen": "https://ottawacitizen.com/feed/",

            # AUSTRALIAN & NEW ZEALAND
            "ABC Australia": "https://www.abc.net.au/news/feed/51120/rss.xml",
            "ABC World": "https://www.abc.net.au/news/feed/52278/rss.xml",
            "ABC Politics": "https://www.abc.net.au/news/feed/52292/rss.xml",
            "ABC Business": "https://www.abc.net.au/news/feed/51892/rss.xml",
            "Sydney Morning Herald": "https://www.smh.com.au/rss/feed.xml",
            "The Age": "https://www.theage.com.au/rss/feed.xml",
            "The Australian": "https://www.theaustralian.com.au/feed/",
            "News.com.au": "https://www.news.com.au/feed/",
            "NZ Herald": "https://www.nzherald.co.nz/rss/",
            "Stuff NZ": "https://www.stuff.co.nz/rss/",
            "Radio New Zealand": "https://www.rnz.co.nz/rss/news.xml",

            # EUROPEAN ENGLISH SOURCES
            "France 24": "https://www.france24.com/en/rss",
            "Deutsche Welle": "https://rss.dw.com/rdf/rss-en-all",
            "DW Europe": "https://rss.dw.com/rdf/rss-en-eu",
            "DW World": "https://rss.dw.com/rdf/rss-en-world",
            "DW Business": "https://rss.dw.com/rdf/rss-en-business",
            "DW Science": "https://rss.dw.com/rdf/rss-en-sci",
            "Euronews": "https://www.euronews.com/rss",
            "EU Observer": "https://euobserver.com/rss",
            "Politico Europe": "https://www.politico.eu/feed/",
            "The Local Europe": "https://www.thelocal.com/feed/",
            "Irish Times": "https://www.irishtimes.com/rss/",
            "Irish Independent": "https://www.independent.ie/rss/",
            "RTÉ News": "https://www.rte.ie/rss/news/",
            "The Journal.ie": "https://www.thejournal.ie/feed/",
            "Dutch News": "https://www.dutchnews.nl/feed/",
            "The Brussels Times": "https://www.brusselstimes.com/feed/",
            "Swiss Info": "https://www.swissinfo.ch/eng/rss/",
            "Anadolu Agency": "https://www.aa.com.tr/en/rss/default?cat=general",

            # ASIAN ENGLISH SOURCES
            "South China Morning Post": "https://www.scmp.com/rss/91/feed",
            "SCMP China": "https://www.scmp.com/rss/2/feed",
            "SCMP Asia": "https://www.scmp.com/rss/4/feed",
            "SCMP World": "https://www.scmp.com/rss/3/feed",
            "SCMP Business": "https://www.scmp.com/rss/18/feed",
            "SCMP Technology": "https://www.scmp.com/rss/36/feed",
            "Japan Times": "https://www.japantimes.co.jp/feed/",
            "The Straits Times": "https://www.straitstimes.com/news/singapore/rss.xml",
            "Straits Times Asia": "https://www.straitstimes.com/news/asia/rss.xml",
            "Straits Times World": "https://www.straitstimes.com/news/world/rss.xml",
            "Straits Times Business": "https://www.straitstimes.com/news/business/rss.xml",
            "Channel News Asia": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3764/rss/",
            "CNA Asia": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3216/rss/",
            "CNA World": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3218/rss/",
            "CNA Business": "https://www.channelnewsasia.com/api/v1/rss-outbound/channel/3214/rss/",
            "Manila Bulletin": "https://mb.com.ph/feed/",
            "Philippine Star": "https://www.philstar.com/rss/headlines",
            "Bangkok Post": "https://www.bangkokpost.com/rss/data/feed.xml",
            "The Nation Thailand": "https://www.nationthailand.com/rss/",
            "Jakarta Post": "https://www.thejakartapost.com/rss/",
            "Korea Herald": "https://www.koreaherald.com/rss/",
            "Korea Times": "https://www.koreatimes.co.kr/rss/",
            "Taiwan News": "https://www.taiwannews.com.tw/rss/",
            "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
            "TOI World": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
            "TOI India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
            "TOI Business": "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
            "TOI Technology": "https://timesofindia.indiatimes.com/rssfeeds/5880659.cms",
            "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/",
            "The Hindu": "https://www.thehindu.com/feeder/default.rss",
            "Indian Express": "https://indianexpress.com/feed/",
            "Mint": "https://www.livemint.com/rss/news",
            "Economic Times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
            "The Print": "https://theprint.in/feed/",
            "Gulf News": "https://gulfnews.com/feed/",
            "Khaleej Times": "https://www.khaleejtimes.com/rss/",
            "The National UAE": "https://www.thenationalnews.com/feed/",
            "Arab News": "https://www.arabnews.com/rss.xml",
            "Jerusalem Post": "https://www.jpost.com/Rss/RssFeedsHeadlines.aspx",
            "Haaretz": "https://www.haaretz.com/rss/",
            "Times of Israel": "https://www.timesofisrael.com/feed/",

            # AFRICAN ENGLISH SOURCES
            "AllAfrica": "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf",
            "Daily Nation Kenya": "https://www.nation.co.ke/rss/",
            "The Star Kenya": "https://www.the-star.co.ke/rss/",
            "Business Day South Africa": "https://www.businesslive.co.za/rss/",
            "Mail & Guardian": "https://mg.co.za/feed/",
            "News24": "https://www.news24.com/feed/",
            "Sowetan Live": "https://www.sowetanlive.co.za/feed/",
            "The Citizen Tanzania": "https://www.thecitizen.co.tz/feed/",
            "Premium Times Nigeria": "https://www.premiumtimesng.com/feed/",
            "Vanguard Nigeria": "https://www.vanguardngr.com/feed/",
            "Punch Nigeria": "https://punchng.com/feed/",
            "This Day Nigeria": "https://www.thisdaylive.com/rss/",
            "The Guardian Nigeria": "https://guardian.ng/feed/",
            "Egypt Independent": "https://www.egyptindependent.com/feed/",
            "Daily News Egypt": "https://dailynewsegypt.com/feed/",

            # LATIN AMERICAN ENGLISH SOURCES
            "Buenos Aires Herald": "https://buenosairesherald.com/feed/",
            "The Rio Times": "https://www.riotimesonline.com/feed/",
            "Mexico News Daily": "https://mexiconewsdaily.com/feed/",
            "Tico Times": "https://ticotimes.net/feed/",
            "Jamaica Observer": "https://www.jamaicaobserver.com/feed/",
            "Trinidad Express": "https://trinidadexpress.com/feed/",

            # TECH NEWS
            "TechCrunch": "https://techcrunch.com/feed/",
            "The Verge": "https://www.theverge.com/rss/index.xml",
            "Wired": "https://www.wired.com/feed/rss",
            "Ars Technica": "https://arstechnica.com/feed/",
            "Mashable": "https://mashable.com/feeds/rss/all",
            "Gizmodo": "https://gizmodo.com/rss",
            "Engadget": "https://www.engadget.com/rss.xml",
            "CNET": "https://www.cnet.com/rss/news/",
            "ZDNet": "https://www.zdnet.com/news/rss.xml",
            "TechRadar": "https://www.techradar.com/rss",
            "Digital Trends": "https://www.digitaltrends.com/feed/",
            "VentureBeat": "https://venturebeat.com/feed/",
            "Recode": "https://www.vox.com/rss/recode/index.xml",
            "TechSpot": "https://www.techspot.com/feed/",
            "Slashdot": "https://rss.slashdot.org/Slashdot/slashdot",
            "The Next Web": "https://thenextweb.com/feed/",
            "Silicon Angle": "https://siliconangle.com/feed/",
            "Computerworld": "https://www.computerworld.com/index.rss",
            "InfoWorld": "https://www.infoworld.com/index.rss",
            "PC World": "https://www.pcworld.com/index.rss",
            "MacRumors": "https://feeds.macrumors.com/MacRumors-All",
            "9to5Mac": "https://9to5mac.com/feed/",
            "Android Central": "https://www.androidcentral.com/feed",
            "XDA Developers": "https://www.xda-developers.com/feed/",
            "TechCrunch Startups": "https://techcrunch.com/startups/feed/",
            "TechCrunch AI": "https://techcrunch.com/ai/feed/",
            "TechCrunch Security": "https://techcrunch.com/security/feed/",

            # BUSINESS & FINANCE
            "Wall Street Journal": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
            "Financial Times": "https://www.ft.com/rss/home",
            "Bloomberg": "https://feeds.bloomberg.com/bloomberg/news.rss",
            "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "CNBC World": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
            "CNBC Business": "https://www.cnbc.com/id/10001147/device/rss/rss.html",
            "CNBC Technology": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
            "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "Investing.com": "https://www.investing.com/rss/news.rss",
            "Yahoo Finance": "https://finance.yahoo.com/news/rss/",
            "Seeking Alpha": "https://seekingalpha.com/feed.xml",
            "The Street": "https://www.thestreet.com/feed/rss/",
            "Business Insider": "https://markets.businessinsider.com/rss/news",
            "Forbes": "https://www.forbes.com/business/feed/",
            "Fortune": "https://fortune.com/feed/",
            "Economist Business": "https://www.economist.com/business/rss.xml",
            "Economist Finance": "https://www.economist.com/finance-and-economics/rss.xml",

            # SCIENCE & ENVIRONMENT
            "Nature": "https://www.nature.com/nature.rss",
            "Science Magazine": "https://www.science.org/feed/rss/all.xml",
            "New Scientist": "https://www.newscientist.com/feed/home/",
            "Scientific American": "https://www.scientificamerican.com/rss/",
            "Science Daily": "https://www.sciencedaily.com/rss/all.xml",
            "Phys.org": "https://phys.org/rss-feed/",
            "Space.com": "https://www.space.com/feeds/all",
            "NASA": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
            "ESA": "https://www.esa.int/feeds/rss/latestnews",
            "National Geographic": "https://www.nationalgeographic.com/rss/",
            "Smithsonian": "https://www.smithsonianmag.com/rss/",
            "Discover Magazine": "https://www.discovermagazine.com/rss",
            "Popular Science": "https://www.popsci.com/rss/",
            "Science News": "https://www.sciencenews.org/feed",
            "Live Science": "https://www.livescience.com/feeds/all",
            "EurekAlert": "https://www.eurekalert.org/rss.xml",
            "Medical News Today": "https://www.medicalnewstoday.com/rss",

            # HEALTH & MEDICINE
            "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx",
            "Healthline": "https://www.healthline.com/health/feed",
            "Mayo Clinic": "https://www.mayoclinic.org/rss/all-health-information-topics",
            "CDC": "https://tools.cdc.gov/api/v2/resources/media/404952.rss",
            "WHO": "https://www.who.int/feeds/entity/csr/don/en/rss.xml",
            "Medical Xpress": "https://medicalxpress.com/rss-feed/",
            "STAT News": "https://www.statnews.com/feed/",
            "Fierce Healthcare": "https://www.fiercehealthcare.com/rss/xml",

            # SPORTS
            "ESPN": "https://www.espn.com/espn/rss/news",
            "ESPN MLB": "https://www.espn.com/espn/rss/mlb/news",
            "ESPN NFL": "https://www.espn.com/espn/rss/nfl/news",
            "ESPN NBA": "https://www.espn.com/espn/rss/nba/news",
            "ESPN NHL": "https://www.espn.com/espn/rss/nhl/news",
            "BBC Sport": "https://feeds.bbci.co.uk/sport/rss.xml",
            "Sky Sports": "https://www.skysports.com/rss/12040",
            "The Athletic": "https://theathletic.com/rss/",
            "Sports Illustrated": "https://www.si.com/rss",
            "CBS Sports": "https://www.cbssports.com/rss/headlines/",
            "Fox Sports": "https://www.foxsports.com/rss",
            "Yahoo Sports": "https://sports.yahoo.com/rss/",
            "Bleacher Report": "https://bleacherreport.com/rss",

            # ENTERTAINMENT & ARTS
            "Variety": "https://variety.com/feed/",
            "Hollywood Reporter": "https://www.hollywoodreporter.com/feed/",
            "Entertainment Weekly": "https://ew.com/feed/",
            "Deadline": "https://deadline.com/feed/",
            "Rolling Stone": "https://www.rollingstone.com/feed/",
            "Pitchfork": "https://pitchfork.com/feed/feed-news/rss",
            "Billboard": "https://www.billboard.com/feed/",
            "MTV News": "https://www.mtv.com/news/feed/",
            "E! Online": "https://www.eonline.com/news/feed",
            "People": "https://people.com/feed/",
            "Us Weekly": "https://www.usmagazine.com/feed/",

            # POLITICS & GOVERNMENT
            "The Hill": "https://thehill.com/news/feed/",
            "Politico": "https://www.politico.com/rss/politicopicks.xml",
            "Real Clear Politics": "https://www.realclearpolitics.com/RSS.xml",
            "FiveThirtyEight": "https://fivethirtyeight.com/politics/feed/",
            "Washington Post Politics": "https://feeds.washingtonpost.com/rss/politics",
            "NYT Politics": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
            "CNN Politics": "https://rss.cnn.com/rss/edition_politics.rss",
            "Fox News Politics": "https://moxie.foxnews.com/google-publisher/politics.xml",
            "NPR Politics": "https://feeds.npr.org/1014/rss.xml",
            "BBC Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
            "The Guardian Politics": "https://www.theguardian.com/politics/rss",

            # WEATHER & DISASTERS
            "Weather Channel": "https://weather.com/rss/",
            "AccuWeather": "https://rss.accuweather.com/rss/liveweather_rss.asp",
            "NOAA": "https://www.noaa.gov/feed/",
            "USGS Earthquake": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.atom",

            # EDUCATION
            "Education Week": "https://www.edweek.org/feed/",
            "Inside Higher Ed": "https://www.insidehighered.com/rss/news",
            "Chronicle of Higher Education": "https://www.chronicle.com/rss/",

            # AUTOMOTIVE
            "Car and Driver": "https://www.caranddriver.com/rss/",
            "Motor Trend": "https://www.motortrend.com/feed/",
            "Autoblog": "https://www.autoblog.com/rss/",
            "Road & Track": "https://www.roadandtrack.com/rss/",

            # TRAVEL
            "Lonely Planet": "https://www.lonelyplanet.com/news/feed",
            "Travel + Leisure": "https://www.travelandleisure.com/feed",
            "Condé Nast Traveler": "https://www.cntraveler.com/feed",

            # FOOD
            "Food Network": "https://www.foodnetwork.com/feed",
            "Bon Appétit": "https://www.bonappetit.com/feed",
            "Eater": "https://www.eater.com/rss/index.xml",

            # FASHION
            "Vogue": "https://www.vogue.com/feed/",
            "Women's Wear Daily": "https://wwd.com/feed/",
            "The Business of Fashion": "https://www.businessoffashion.com/rss",

            # REAL ESTATE
            "Realtor.com": "https://www.realtor.com/news/feed/",
            "Zillow": "https://www.zillow.com/blog/feed/",

            # LOCAL US NEWS
            "NY1": "https://ny1.com/nyc/all-boroughs/feed",
            "Chicago Sun-Times": "https://chicago.suntimes.com/feed/",
            "Philadelphia Inquirer": "https://www.inquirer.com/feed/",
            "Detroit Free Press": "httpswww.freep.com/rss/",
            "St. Louis Post-Dispatch": "https://www.stltoday.com/rss/",
            "Minneapolis Star Tribune": "https://www.startribune.com/rss/",
            "Portland Oregonian": "https://www.oregonlive.com/rss/",
            "Las Vegas Review-Journal": "https://www.reviewjournal.com/feed/",
            "Orlando Sentinel": "https://www.orlandosentinel.com/feed/",
            "Tampa Bay Times": "https://www.tampabay.com/feed/",
            "Salt Lake Tribune": "https://www.sltrib.com/rss/",
            "Kansas City Star": "https://www.kansascity.com/rss/",
            "Indianapolis Star": "https://www.indystar.com/rss/",
            "Charlotte Observer": "https://www.charlotteobserver.com/rss/",
            "Raleigh News & Observer": "https://www.newsobserver.com/rss/",
            "Nashville Tennessean": "https://www.tennessean.com/rss/",
            "Austin American-Statesman": "https://www.statesman.com/rss/",
            "San Antonio Express-News": "https://www.expressnews.com/rss/",
            "Sacramento Bee": "https://www.sacbee.com/rss/",
            "San Diego Union-Tribune": "https://www.sandiegouniontribune.com/rss/",
            "Seattle Post-Intelligencer": "https://www.seattlepi.com/rss/",
            "Boston Herald": "https://www.bostonherald.com/feed/",
            "NY Daily News": "https://www.nydailynews.com/feed/",

            # SPECIALIZED & TRADE PUBLICATIONS
            "Advertising Age": "https://adage.com/rss",
            "PR Week": "https://www.prweek.com/rss",
            "Law360": "https://www.law360.com/rss",
            "Legal News": "https://www.law.com/rss/",
            "MedPage Today": "https://www.medpagetoday.com/rss/",
            "Healthcare IT News": "https://www.healthcareitnews.com/rss",
            "Modern Healthcare": "https://www.modernhealthcare.com/rss",
            "Construction Dive": "https://www.constructiondive.com/rss/",
            "Retail Dive": "https://www.retaildive.com/rss/",
            "Marketing Dive": "https://www.marketingdive.com/rss/",
            "HR Dive": "https://www.hrdive.com/rss/",
            "Supply Chain Dive": "https://www.supplychaindive.com/rss/",
            "Energy News": "https://www.energy-dive.com/rss/",
            "Utility Dive": "https://www.utilitydive.com/rss/",
            "BioPharma Dive": "https://www.biopharmadive.com/rss/",
            "MedTech Dive": "https://www.medtechdive.com/rss/",
            "Transport Dive": "https://www.transportdive.com/rss/",
            "Food Dive": "https://www.fooddive.com/rss/",
            "Hotel News Now": "https://www.hotelnewsnow.com/RSS",
            "Travel Weekly": "https://www.travelweekly.com/RSS",
            "Skift": "https://skift.com/feed/",
            "Pharma Times": "https://www.pharmatimes.com/rss/",
            "Clinical Trials Arena": "https://www.clinicaltrialsarena.com/rss/",
            "Medical Device Network": "https://www.medicaldevice-network.com/rss/",
            "Mining Technology": "https://www.mining-technology.com/rss/",
            "Army Technology": "https://www.army-technology.com/rss/",
            "Naval Technology": "https://www.naval-technology.com/rss/",
            "Airforce Technology": "https://www.airforce-technology.com/rss/",
            "Space News": "https://spacenews.com/feed/",
            "Satellite Today": "https://www.satellitetoday.com/feed/",
            "Via Satellite": "https://www.satellitetoday.com/feed/",
            "Broadband News": "https://www.broadbandnews.com/feed/",
            "Telecoms News": "https://www.telecoms.com/feed/",
            "Mobile World Live": "https://www.mobileworldlive.com/feed/",
            "Light Reading": "https://www.lightreading.com/rss",
            "Fierce Telecom": "https://www.fiercetelecom.com/rss/xml",
            "Fierce Wireless": "https://www.fiercewireless.com/rss/xml",
            "Fierce Video": "https://www.fiercevideo.com/rss/xml",
            "Fierce Electronics": "https://www.fierceelectronics.com/rss/xml",
            "Electronic Design": "https://www.electronicdesign.com/rss",
            "EE Times": "https://www.eetimes.com/feed/",
            "EDN": "https://www.edn.com/feed/",
            "Embedded": "https://www.embedded.com/rss-feed/",
            "Design News": "https://www.designnews.com/rss",
            "Machine Design": "https://www.machinedesign.com/rss",
            "Power Electronics": "https://www.powerelectronics.com/rss",
            "Microwave Journal": "https://www.microwavejournal.com/rss",
            "IEEE Spectrum": "https://spectrum.ieee.org/rss",
            "Physics World": "https://physicsworld.com/feed/",
            "Optics.org": "https://optics.org/rss",
            "Laser Focus World": "https://www.laserfocusworld.com/rss",
            "Photonics.com": "https://www.photonics.com/rss",
            "Biophotonics": "https://www.biophotonics.com/rss",
            "Vision Systems Design": "https://www.vision-systems.com/rss",
            "Control Engineering": "https://www.controleng.com/rss",
            "Plant Engineering": "https://www.plantengineering.com/rss",
            "Consulting-Specifying Engineer": "https://www.csemag.com/rss",
            "Food Engineering": "https://www.foodengineeringmag.com/rss",
            "Powder & Bulk Solids": "https://www.powderbulksolids.com/rss",
            "Chemical Processing": "https://www.chemicalprocessing.com/rss",
            "Pharmaceutical Technology": "https://www.pharmtech.com/rss",
            "BioPharm International": "https://www.biopharminternational.com/rss",
            "Laboratory Equipment": "https://www.laboratoryequipment.com/rss",
            "R&D Magazine": "https://www.rdmag.com/rss",
            "Scientific Computing": "https://www.scientificcomputing.com/rss",
            "GenomeWeb": "https://www.genomeweb.com/rss",
            "Bio-IT World": "https://www.bio-itworld.com/rss",
            "Drug Discovery News": "https://www.drugdiscoverynews.com/rss",
            "Science Business": "https://sciencebusiness.net/rss",
            "Research Europe": "https://www.researchresearch.com/rss",
            "Times Higher Education": "https://www.timeshighereducation.com/rss",
            "Inside Science": "https://www.insidescience.org/rss",
            "Physics Today": "https://physicstoday.scitation.org/rss",
            "Chemistry World": "https://www.chemistryworld.com/rss",
            "Materials Today": "https://www.materialstoday.com/rss",
            "Nanowerk": "https://www.nanowerk.com/rss",
            "AZoNano": "https://www.azonano.com/rss",
            "AZoM": "https://www.azom.com/rss",
            "AZoCleantech": "https://www.azocleantech.com/rss",
            "AZoRobotics": "https://www.azorobotics.com/rss",
            "AZoAI": "https://www.azoai.com/rss",
            "AZoQuantum": "https://www.azoquantum.com/rss",
            "AZoOptics": "https://www.azooptics.com/rss",
            "AZoSensors": "https://www.azosensors.com/rss",
            "AZoNetwork": "https://www.azonetwork.com/rss",
        }


    def fetch_from_all_rss_feeds_parallel(self) -> List[Dict]:
        """Fetch from ALL 463 RSS feeds in parallel with optimized settings"""
        rss_feeds = self._get_global_news_sources()  # This keeps all 463 sources
        self.logger.info(f"📰 Fetching from ALL {len(rss_feeds)} RSS feeds in parallel...")
        
        all_articles = []
        successful_feeds = 0
        failed_feeds = 0
        
        # INCREASED parallel workers to handle more sources simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:  # Increased from 20 to 50
            # Submit all 463 tasks
            future_to_feed = {
                executor.submit(self._fetch_single_rss_feed, name, url): name 
                for name, url in rss_feeds.items()
            }
            
            # Process completed tasks as they finish
            completed = 0
            for future in concurrent.futures.as_completed(future_to_feed):
                name = future_to_feed[future]
                completed += 1
                
                try:
                    articles = future.result(timeout=self.timeout_seconds + 2)
                    if articles:
                        all_articles.extend(articles)
                        successful_feeds += 1
                        if successful_feeds % 20 == 0:  # Log progress every 20 successful feeds
                            self.logger.info(f"📊 Progress: {completed}/{len(rss_feeds)} feeds, {successful_feeds} successful, {len(all_articles)} articles")
                except concurrent.futures.TimeoutError:
                    failed_feeds += 1
                    self.logger.debug(f"⏰ {name}: Timeout")
                except Exception as e:
                    failed_feeds += 1
                    self.logger.debug(f"⚠️ {name}: {str(e)[:50]}...")
        
        self.logger.info(f"✅ RSS COMPLETE: {len(all_articles)} articles from {successful_feeds}/{len(rss_feeds)} feeds ({failed_feeds} failed)")
        return all_articles

    def _fetch_single_rss_feed(self, name: str, url: str) -> List[Dict]:
        """Fetch articles from a single RSS feed with timeout - OPTIMIZED VERSION"""
        articles = []
        try:
            # Use a shorter timeout for individual feeds
            response = requests.get(url, timeout=6)  # Reduced from 8s to 6s
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                
                # Process only recent articles to save time
                for entry in feed.entries[:15]:  # Reduced from 20 to 15
                    if self._is_recent(entry.get('published_parsed')):
                        summary = entry.get('summary', '')
                        if '<' in summary:
                            # Use faster HTML stripping
                            summary = re.sub('<[^<]+?>', '', summary)
                        
                        article = {
                            'title': entry.get('title', 'No Title'),
                            'description': summary[:300],  # Limit description length
                            'content': summary[:500],      # Limit content length
                            'url': entry.get('link'),
                            'publishedAt': datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat() if entry.get('published_parsed') else datetime.utcnow().isoformat(),
                            'source': {'name': name},
                            'urlToImage': ''
                        }
                        articles.append(article)
                return articles
            else:
                return []  # Silently skip failed HTTP requests
        except Exception:
            return []  # Silently skip all errors for speed

    def _fetch_with_timeout(self, fetch_function, source_name, *args, **kwargs):
        """Execute a fetch function with timeout protection"""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_function, *args, **kwargs)
                articles = future.result(timeout=self.timeout_seconds)
                return articles
        except concurrent.futures.TimeoutError:
            self.logger.warning(f"⏰ TIMEOUT: {source_name} exceeded {self.timeout_seconds} seconds")
            return []
        except Exception as e:
            self.logger.error(f"❌ {source_name} error: {e}")
            return []

    def fetch_from_newsdata_today(self) -> List[Dict]:
        """Fetch ALL today's news from NewsData.io with timeout protection"""
        articles: List[Dict] = []
        try:
            start_time = time.time()
            
            def fetch_newsdata():
                url = "https://newsdata.io/api/1/news"
                api_key = self.config.NEWS_API_KEYS.get('newsdata')
                if not api_key:
                    return []

                params = {'apikey': api_key, 'language': 'en', 'timeframe': 48, "category": ""}
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
                if response.status_code == 200:
                    news_articles = []
                    for article in response.json().get('results', []):
                        news_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('link', ''),
                            'publishedAt': article.get('pubDate', ''),
                            'source': {'name': article.get('source_id', 'NewsData.io')},
                            'urlToImage': article.get('image_url', '')
                        })
                    return news_articles
                else:
                    self.logger.warning(f"⚠️ NewsData.io HTTP {response.status_code}")
                    return []
            
            articles = self._fetch_with_timeout(fetch_newsdata, "NewsData.io")
            elapsed_time = time.time() - start_time
            
            if articles:
                self.logger.info(f"✅ NewsData.io: {len(articles)} articles in {elapsed_time:.1f}s")
            else:
                self.logger.info(f"⏩ NewsData.io: Skipped (timeout or error)")
                
        except Exception as e:
            self.logger.error(f"❌ NewsData.io error: {e}")
        return articles

    def fetch_from_newsapi_today(self) -> List[Dict]:
        """Fetch ALL today's news from NewsAPI with timeout protection"""
        articles: List[Dict] = []
        try:
            start_time = time.time()
            
            def fetch_newsapi():
                url = "https://newsapi.org/v2/top-headlines"
                api_key = self.config.NEWS_API_KEYS.get("newsapi")
                if not api_key:
                    return []

                params = {
                    "category": "technology",
                    "language": "en",
                    "pageSize": 100,
                    "apiKey": api_key,
                }
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
                if response.status_code == 200:
                    news_articles = []
                    raw_articles = response.json().get("articles", [])
                    for a in raw_articles:
                        news_articles.append({
                            "title": a.get("title", "").strip(),
                            "description": a.get("description", "").strip(),
                            "url": a.get("url"),
                            "source": {'name': a.get("source", {}).get("name", "NewsAPI")},
                            "publishedAt": a.get("publishedAt", ""),
                        })
                    return news_articles
                else:
                    self.logger.warning(f"⚠️ NewsAPI HTTP {response.status_code}")
                    return []
            
            articles = self._fetch_with_timeout(fetch_newsapi, "NewsAPI")
            elapsed_time = time.time() - start_time
            
            if articles:
                self.logger.info(f"✅ NewsAPI: {len(articles)} articles in {elapsed_time:.1f}s")
            else:
                self.logger.info(f"⏩ NewsAPI: Skipped (timeout or error)")
                
        except Exception as e:
            self.logger.error(f"❌ NewsAPI fetch error: {e}")
        return articles

    def fetch_from_gnews_today(self) -> List[Dict]:
        """Fetch ALL today's news from GNews with timeout protection"""
        articles: List[Dict] = []
        try:
            start_time = time.time()
            
            def fetch_gnews():
                url = "https://gnews.io/api/v4/top-headlines"
                api_key = self.config.NEWS_API_KEYS.get('gnews')
                if not api_key:
                    return []

                params = {
                    'topic': 'technology',
                    'lang': 'en',
                    'max': 100,
                    'apikey': api_key
                }
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
                if response.status_code == 200:
                    return response.json().get('articles', [])
                else:
                    self.logger.warning(f"⚠️ GNews HTTP {response.status_code}")
                    return []
            
            articles = self._fetch_with_timeout(fetch_gnews, "GNews")
            elapsed_time = time.time() - start_time
            
            if articles:
                self.logger.info(f"✅ GNews: {len(articles)} articles in {elapsed_time:.1f}s")
            else:
                self.logger.info(f"⏩ GNews: Skipped (timeout or error)")
                
        except Exception as e:
            self.logger.error(f"❌ GNews error: {e}")
        return articles

    def fetch_all_news(self) -> List[Dict]:
        """
        The main orchestrator. Fetches from all sources with timeout protection.
        """
        all_raw_articles = []
        self.logger.info(f"🌐 Starting news collection with {self.timeout_seconds}s timeout per source...")

        sources_to_fetch = [
            self.fetch_from_newsdata_today,
            self.fetch_from_newsapi_today,
            self.fetch_from_gnews_today,
            self.fetch_from_all_rss_feeds_parallel,  # Use the parallel version
        ]

        for fetch_function in sources_to_fetch:
            try:
                self.logger.info(f"🔄 Fetching from {fetch_function.__name__}...")
                start_time = time.time()
                new_articles = fetch_function()
                elapsed_time = time.time() - start_time
                
                if new_articles:
                    all_raw_articles.extend(new_articles)
                    self.logger.info(f"✅ {fetch_function.__name__}: {len(new_articles)} articles in {elapsed_time:.1f}s")
                else:
                    self.logger.warning(f"⏩ {fetch_function.__name__}: No articles in {elapsed_time:.1f}s")
                    
            except Exception as e:
                self.logger.error(f"❌ {fetch_function.__name__} failed: {e}")
                continue  # Move to next source

        # Simple URL-based deduplication
        seen_urls = set()
        unique_articles = []
        for article in all_raw_articles:
            url = article.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)

        unique_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
        
        self.logger.info(f"📊 FINAL: {len(unique_articles)} unique articles from {len(all_raw_articles)} total")
        
        return unique_articles