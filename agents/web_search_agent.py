# agents/web_search_agent.py
from .base import BaseAgent
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import os
import json
import time
from datetime import datetime, timedelta
import hashlib
from gemini_utils import embed_text
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
import pickle

# Import the approved domains configuration
from config.approved_domains import APPROVED_DOMAINS, ALL_APPROVED_DOMAINS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebSearchAgent")

class WebSearchAgent(BaseAgent):
    """Agent responsible for conducting domain-restricted web searches."""
    
    def __init__(self, cache_dir="web_cache"):
        """Initialize the web search agent with caching capabilities."""
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
        
        # Load existing cache index or create a new one
        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.pkl")
        self.cache_index = self._load_cache_index()
        
        # Headers to mimic a browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        logger.info("WebSearchAgent initialized successfully")
    
    def _load_cache_index(self) -> Dict:
        """Load the cache index or create a new one if it doesn't exist."""
        if os.path.exists(self.cache_index_path):
            try:
                with open(self.cache_index_path, 'rb') as f:
                    cache_index = pickle.load(f)
                logger.info(f"Cache index loaded with {len(cache_index)} entries")
                return cache_index
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                return {}
        else:
            logger.info("Creating new cache index")
            return {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        with open(self.cache_index_path, 'wb') as f:
            pickle.dump(self.cache_index, f)
        logger.info(f"Cache index saved with {len(self.cache_index)} entries")
    
    def validate_domain(self, url: str) -> bool:
        """
        Check if the URL belongs to an approved domain.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if domain is approved, False otherwise
        """
        # Parse the URL and extract domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Construct path-inclusive domain for matching
        path_domain = domain + parsed_url.path
        
        # Check against all approved domains
        for approved_domain in ALL_APPROVED_DOMAINS:
            parsed_approved = urlparse(f"https://{approved_domain}")
            approved_netloc = parsed_approved.netloc
            
            # Remove 'www.' prefix if present
            if approved_netloc.startswith('www.'):
                approved_netloc = approved_netloc[4:]
                
            approved_path = approved_netloc + parsed_approved.path
            
            # Match both domain and path components
            if domain == approved_netloc or path_domain.startswith(approved_path):
                logger.info(f"URL validated: {url} matches approved domain {approved_domain}")
                return True
        
        logger.warning(f"URL rejected: {url} does not match any approved domain")
        return False
    
    def get_best_topic_category(self, query: str) -> str:
        """
        Determine the most relevant topic category for a given query.
        
        Args:
            query: User query string
            
        Returns:
            str: Best matching topic category
        """
        # Convert query to lowercase for matching
        query_lower = query.lower()
        
        # Simple keyword-based matching
        topic_scores = {}
        
        for topic, domains in APPROVED_DOMAINS.items():
            # Initialize score for this topic
            score = 0
            
            # Convert topic words to individual matching terms
            topic_words = topic.replace("_", " ").lower().split()
            
            # Award points for topic words appearing in query
            for word in topic_words:
                if word in query_lower:
                    score += 2
            
            # Check for similar words or partial matches
            for word in topic_words:
                if len(word) > 4:  # Only check substantial words
                    for q_word in query_lower.split():
                        # Fuzzy matching - if query word contains most of topic word
                        if len(word) > 3 and word[:3] in q_word:
                            score += 0.5
            
            # Additional domain-specific heuristics
            if topic == "wright_brothers" and ("aviation" in query_lower or "airplane" in query_lower or "flight" in query_lower):
                score += 1.5
            elif topic == "education_sri_lanka" and ("education" in query_lower or "school" in query_lower or "sri lanka" in query_lower):
                score += 1.5
            elif topic == "mahaweli_development" and ("development" in query_lower or "irrigation" in query_lower or "river" in query_lower):
                score += 1.5
            elif topic == "marie_antoinette" and ("france" in query_lower or "revolution" in query_lower or "cake" in query_lower):
                score += 1.5
            elif topic == "adolf_hitler" and ("nazi" in query_lower or "germany" in query_lower or "ww2" in query_lower or "world war" in query_lower):
                score += 1.5
                
            topic_scores[topic] = score
        
        # Get topic with highest score, default to first topic if all scores are 0
        best_topic = max(topic_scores.items(), key=lambda x: x[1])
        
        # If best score is 0, check if any words in query match with any domain
        if best_topic[1] == 0:
            logger.info(f"No direct topic match found for query: {query}. Using default.")
            return list(APPROVED_DOMAINS.keys())[0]
        
        logger.info(f"Best topic for '{query}': {best_topic[0]} (score: {best_topic[1]})")
        return best_topic[0]
    
    def extract_domain(self, url: str) -> str:
        """Extract the domain from a URL for simpler comparison."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain
    
    def _get_cache_key(self, url: str) -> str:
        """Generate a unique cache key for a URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if a cache entry is still valid based on its timestamp."""
        # Set cache validity to 7 days
        cache_valid_seconds = 7 * 24 * 60 * 60
        
        # Check if the entry has a timestamp and is not too old
        if 'timestamp' in cache_entry:
            age = time.time() - cache_entry['timestamp']
            return age < cache_valid_seconds
        
        return False
    
    def fetch_url_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL with domain validation and caching.
        
        Args:
            url: URL to fetch
            
        Returns:
            Optional[str]: HTML content of the page or None if failed/invalid
        """
        # Validate the URL domain before fetching
        if not self.validate_domain(url):
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.html")
        
        # If we have a valid cache entry, return cached content
        if cache_key in self.cache_index and self._is_cache_valid(self.cache_index[cache_key]):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Cache hit for URL: {url}")
                return content
            except Exception as e:
                logger.warning(f"Cache read failed for {url}: {e}")
                # Continue to fetch if cache read fails
        
        # Fetch the content
        try:
            logger.info(f"Fetching URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Store in cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Update cache index
            self.cache_index[cache_key] = {
                'url': url,
                'timestamp': time.time(),
                'size': len(response.text)
            }
            self._save_cache_index()
            
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract main text content from HTML.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            str: Extracted text content
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator='\n')
            
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def chunk_text(self, text: str, max_chars: int = 1500) -> List[Dict]:
        """
        Divide text into semantic chunks of appropriate size.
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List[Dict]: List of chunks with text and metadata
        """
        # Split text into paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the max size, store the current chunk
            if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Convert to dictionary format with metadata
        return [{"text": chunk, "metadata": {"source": "web", "chunk_id": i}} 
                for i, chunk in enumerate(chunks)]
    
    def score_web_results(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Score web search results based on relevance to query.
        
        Args:
            query: Original user query
            chunks: List of text chunks with metadata
            
        Returns:
            List[Dict]: Same chunks with added relevance scores
        """
        if not chunks:
            return []
        
        # Calculate query embedding for semantic similarity
        query_embedding = np.array(embed_text(query), dtype="float32")
        
        # Process each chunk
        for chunk in chunks:
            # Get text embedding
            text_embedding = np.array(embed_text(chunk["text"]), dtype="float32")
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, text_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
            )
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            text_words = set(chunk["text"].lower().split())
            keyword_overlap = len(query_words.intersection(text_words)) / max(len(query_words), 1)
            
            # Combined score (70% semantic, 30% keyword)
            combined_score = (similarity * 0.7) + (keyword_overlap * 0.3)
            
            # Add scores to chunk metadata
            chunk["semantic_score"] = float(similarity)
            chunk["keyword_score"] = float(keyword_overlap)
            chunk["combined_score"] = float(combined_score)
            
            # Calculate confidence similar to RetrieverAgent
            if combined_score > 0.8:
                confidence = 0.9  # Very high confidence
            elif combined_score > 0.6:
                confidence = 0.7  # High confidence
            elif combined_score > 0.4:
                confidence = 0.5  # Medium confidence
            elif combined_score > 0.2:
                confidence = 0.3  # Low confidence
            else:
                confidence = 0.1  # Very low confidence
                
            chunk["confidence"] = float(confidence)
        
        # Sort by combined score
        return sorted(chunks, key=lambda x: x["combined_score"], reverse=True)
    
    def search_web(self, query: str, topic_category: str = None, max_results: int = 3) -> List[Dict]:
        """
        Perform a web search on approved domains based on query.
        
        Args:
            query: User query
            topic_category: Specific topic to search or None for auto-detection
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Ranked and processed web search results
        """
        logger.info(f"Performing web search for query: '{query}'")
        
        # Determine topic category if not provided
        if not topic_category:
            topic_category = self.get_best_topic_category(query)
        
        # Get approved domains for this category
        approved_urls = APPROVED_DOMAINS.get(topic_category, [])
        if not approved_urls:
            logger.warning(f"No approved URLs found for topic: {topic_category}")
            return []
        
        logger.info(f"Searching topic '{topic_category}' with {len(approved_urls)} approved URLs")
        
        all_chunks = []
        for url in approved_urls:
            # Ensure URL has scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Fetch URL content
            html_content = self.fetch_url_content(url)
            if not html_content:
                continue
                
            # Extract text
            text_content = self.extract_text_from_html(html_content)
            if not text_content:
                continue
                
            # Chunk the text
            chunks = self.chunk_text(text_content)
            
            # Add URL to metadata
            for chunk in chunks:
                chunk["metadata"]["url"] = url
                chunk["metadata"]["topic"] = topic_category
            
            all_chunks.extend(chunks)
        
        # Score and rank chunks
        scored_chunks = self.score_web_results(query, all_chunks)
        
        # Return top results
        top_results = scored_chunks[:max_results]
        logger.info(f"Returning {len(top_results)} web search results")
        
        return top_results
    
    def run(self, query: str, query_analysis: Dict = None, max_results: int = 3) -> List[Dict]:
        """
        Run the web search process based on user query.
        
        Args:
            query: User query string
            query_analysis: Analysis from QueryAnalyzerAgent (optional)
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Processed web search results
        """
        print(f"🌐 Searching web for information on: '{query}'")
        
        # Use query analysis if provided
        topic_category = None
        if query_analysis and 'entities' in query_analysis:
            # Try to match entities to topic categories
            for entity in query_analysis['entities']:
                for topic in APPROVED_DOMAINS.keys():
                    if entity.lower() in topic.replace('_', ' ').lower():
                        topic_category = topic
                        break
                if topic_category:
                    break
        
        # Perform search
        search_results = self.search_web(query, topic_category, max_results)
        
        if not search_results:
            print("❌ No relevant web results found.")
            return []
        
        print(f"✅ Found {len(search_results)} relevant web results.")
        return search_results