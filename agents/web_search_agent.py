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
import asyncio
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, Summary
import threading

# Import the approved domains configuration
from config.approved_domains import APPROVED_DOMAINS, ALL_APPROVED_DOMAINS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebSearchAgent")

class WebSearchMetrics:
    """Class for tracking and analyzing web search performance metrics."""
    
    def __init__(self, metrics_dir="web_metrics"):
        """Initialize metrics tracking system."""
        self.metrics_dir = metrics_dir
        
        # Create metrics directory if it doesn't exist
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
            logger.info(f"Created metrics directory: {self.metrics_dir}")
        
        # Initialize metrics storage
        self.request_times = []  # Store response times
        self.topic_accuracy = {}  # Track topic classification accuracy
        self.cache_hits = 0
        self.cache_misses = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.source_distribution = defaultdict(int)
        self.daily_metrics = {}
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
        
        # Initialize counters
        self.search_counter = Counter('web_searches_total', 'Total number of web searches')
        self.cache_hit_counter = Counter('cache_hits_total', 'Total cache hits')
        self.domain_counter = Counter('domains_accessed', 'Domains accessed', ['domain'])
        
        # Initialize histograms
        self.request_time_hist = Histogram('request_duration_seconds', 
                                          'Web request duration in seconds',
                                          buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
        
        # Initialize gauges
        self.success_rate_gauge = Gauge('web_search_success_rate', 'Success rate of web searches')
        
        # Load previous metrics if they exist
        self.load_metrics()
        
        logger.info("Web search metrics tracking initialized")
    
    def record_request_time(self, url: str, duration: float) -> None:
        """Record the response time for a web request."""
        with self.lock:
            self.request_times.append({"url": url, "duration": duration, "timestamp": time.time()})
            self.request_time_hist.observe(duration)
            logger.debug(f"Recorded request time for {url}: {duration:.2f}s")
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self.lock:
            self.cache_hits += 1
            self.cache_hit_counter.inc()
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self.lock:
            self.cache_misses += 1
    
    def record_request_result(self, url: str, success: bool) -> None:
        """Record the result of a web request."""
        with self.lock:
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            domain = urlparse(url).netloc
            self.domain_counter.labels(domain=domain).inc()
            
            # Update success rate gauge
            total = self.successful_requests + self.failed_requests
            if total > 0:
                self.success_rate_gauge.set(self.successful_requests / total)
    
    def record_topic_classification(self, query: str, predicted_topic: str, actual_topic: str = None) -> None:
        """Record topic classification result for accuracy tracking."""
        with self.lock:
            self.topic_accuracy[query] = {
                "predicted": predicted_topic,
                "actual": actual_topic,
                "timestamp": time.time()
            }
    
    def record_source_used(self, source_type: str) -> None:
        """Record which source was used in the final response."""
        with self.lock:
            self.source_distribution[source_type] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Calculate the cache hit rate."""
        with self.lock:
            total = self.cache_hits + self.cache_misses
            if total == 0:
                return 0
            return self.cache_hits / total
    
    def get_average_request_time(self) -> float:
        """Calculate the average request time."""
        with self.lock:
            if not self.request_times:
                return 0
            return sum(item["duration"] for item in self.request_times) / len(self.request_times)
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of web requests."""
        with self.lock:
            total = self.successful_requests + self.failed_requests
            if total == 0:
                return 0
            return self.successful_requests / total
    
    def get_source_distribution(self) -> Dict[str, int]:
        """Get the distribution of sources used in responses."""
        with self.lock:
            return dict(self.source_distribution)
    
    def save_metrics(self) -> None:
        """Save the current metrics to disk."""
        with self.lock:
            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
            
            # Prepare metrics data
            metrics_data = {
                "timestamp": time.time(),
                "cache_hit_rate": self.get_cache_hit_rate(),
                "average_request_time": self.get_average_request_time(),
                "success_rate": self.get_success_rate(),
                "source_distribution": self.get_source_distribution(),
                "total_searches": self.successful_requests + self.failed_requests,
                "total_cache_hits": self.cache_hits,
                "total_cache_misses": self.cache_misses
            }
            
            # Save to file
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"Metrics saved to {metrics_file}")
    
    def load_metrics(self) -> None:
        """Load metrics from the most recent file if it exists."""
        try:
            # Find most recent metrics file
            metric_files = [f for f in os.listdir(self.metrics_dir) if f.startswith("metrics_") and f.endswith(".json")]
            if not metric_files:
                return
                
            most_recent = max(metric_files)
            metrics_file = os.path.join(self.metrics_dir, most_recent)
            
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                
            # Initialize with previous values
            self.cache_hits = metrics_data.get("total_cache_hits", 0)
            self.cache_misses = metrics_data.get("total_cache_misses", 0)
            self.successful_requests = metrics_data.get("total_searches", 0) * metrics_data.get("success_rate", 1)
            self.failed_requests = metrics_data.get("total_searches", 0) - self.successful_requests
            
            for source, count in metrics_data.get("source_distribution", {}).items():
                self.source_distribution[source] = count
                
            logger.info(f"Loaded metrics from {metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to load previous metrics: {e}")

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
        
        # Initialize metrics tracking
        self.metrics = WebSearchMetrics()
        
        # Initialize topic embeddings cache
        self.topic_embeddings = {}
        
        # Initialize async session - will be created on first use
        self.session = None
        
        # Pre-compute topic descriptions for embedding similarity
        self.topic_descriptions = {
            "wright_brothers": "The Wright brothers were American aviation pioneers who invented, built, and flew the world's first successful motor-operated airplane. Their first flight was in December 1903.",
            "education_sri_lanka": "Education in Sri Lanka has a history influenced by British colonial rule, missionary schools, and recent development programs focusing on accessibility and quality.",
            "mahaweli_development": "The Mahaweli Development Programme is Sri Lanka's largest irrigation and hydropower project, started in 1970 to develop agriculture and provide electricity throughout the country.",
            "marie_antoinette": "Marie Antoinette was the last Queen of France before the French Revolution, known for the phrase 'Let them eat cake' although historians debate whether she actually said this.",
            "adolf_hitler": "Adolf Hitler was the leader of Nazi Germany from 1933 to 1945 who initiated World War II in Europe and was central to the Holocaust genocide."
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
    
    def get_best_topic_category(self, query: str) -> Dict:
        """
        Determine the most relevant topic category for a given query using both
        keyword matching and embedding similarity.
        
        Args:
            query: User query string
            
        Returns:
            Dict: Best matching topic info with category, confidence, and approach
        """
        # Track execution time for metrics
        start_time = time.time()
        
        # Convert query to lowercase for matching
        query_lower = query.lower()
        
        # APPROACH 1: Simple keyword-based matching (from existing implementation)
        topic_keyword_scores = {}
        
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
                
            topic_keyword_scores[topic] = score
        
        # Normalize keyword scores to 0-1 range
        max_keyword_score = max(topic_keyword_scores.values()) if topic_keyword_scores else 1
        if max_keyword_score > 0:
            normalized_keyword_scores = {t: s/max_keyword_score for t, s in topic_keyword_scores.items()}
        else:
            normalized_keyword_scores = {t: 0 for t in topic_keyword_scores}

        # APPROACH 2: Embedding-based semantic similarity
        topic_embedding_scores = {}
        
        # Get query embedding
        query_embedding = np.array(embed_text(query), dtype="float32")
        
        # Compare with each topic description
        for topic, description in self.topic_descriptions.items():
            # Get topic embedding (cached for efficiency)
            if topic not in self.topic_embeddings:
                self.topic_embeddings[topic] = np.array(embed_text(description), dtype="float32")
            
            topic_embedding = self.topic_embeddings[topic]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, topic_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(topic_embedding)
            )
            
            topic_embedding_scores[topic] = float(similarity)
        
        # COMBINED APPROACH: Weight keyword and embedding scores
        # Can adjust these weights based on performance
        keyword_weight = 0.4
        embedding_weight = 0.6
        
        combined_scores = {}
        for topic in APPROVED_DOMAINS.keys():
            keyword_score = normalized_keyword_scores.get(topic, 0)
            embedding_score = topic_embedding_scores.get(topic, 0)
            
            combined_scores[topic] = (keyword_score * keyword_weight) + (embedding_score * embedding_weight)
        
        # Get topic with highest combined score
        best_topic, best_score = max(combined_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on margin between best and second-best
        scores_list = sorted(combined_scores.values(), reverse=True)
        if len(scores_list) > 1:
            margin = scores_list[0] - scores_list[1]  # Difference between best and second-best
            confidence = min(0.5 + margin, 0.95)  # Convert margin to confidence score
        else:
            confidence = 0.5  # Default confidence
        
        # If best score is very low, lower confidence
        if best_score < 0.3:
            confidence = best_score + 0.2
        
        # Record performance metrics
        self.metrics.record_topic_classification(query, best_topic)
        
        # Log the result
        duration = time.time() - start_time
        logger.info(f"Topic classification for '{query}': {best_topic} (score: {best_score:.2f}, confidence: {confidence:.2f}, time: {duration:.3f}s)")
        
        return {
            "topic": best_topic,
            "confidence": confidence,
            "score": best_score,
            "keyword_score": normalized_keyword_scores.get(best_topic, 0),
            "embedding_score": topic_embedding_scores.get(best_topic, 0),
            "duration_seconds": duration
        }
        
    async def _create_session(self):
        """Create an aiohttp session for async requests if it doesn't exist."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
            logger.info("Created aiohttp session for async requests")
        return self.session
        
    async def async_fetch_url(self, url: str) -> Tuple[str, Optional[str]]:
        """
        Asynchronously fetch content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple[str, Optional[str]]: Tuple of (url, content or None if failed)
        """
        # Validate the URL domain
        if not self.validate_domain(url):
            return url, None
        
        # Check cache first
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.html")
        
        # If we have a valid cache entry, return cached content
        if cache_key in self.cache_index and self._is_cache_valid(self.cache_index[cache_key]):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Cache hit for URL: {url}")
                self.metrics.record_cache_hit()
                return url, content
            except Exception as e:
                logger.warning(f"Cache read failed for {url}: {e}")
                # Continue to fetch if cache read fails

        # Not in cache or cache invalid, fetch the content
        try:
            # Ensure we have a session
            session = await self._create_session()
            
            start_time = time.time()
            logger.info(f"Fetching URL asynchronously: {url}")
            
            # Use a timeout to avoid hanging on slow servers
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    duration = time.time() - start_time
                    
                    # Store in cache
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Update cache index
                    self.cache_index[cache_key] = {
                        'url': url,
                        'timestamp': time.time(),
                        'size': len(content)
                    }
                    
                    # Record metrics
                    self.metrics.record_request_time(url, duration)
                    self.metrics.record_request_result(url, success=True)
                    
                    return url, content
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    self.metrics.record_request_result(url, success=False)
                    return url, None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching {url}")
            self.metrics.record_request_result(url, success=False)
            return url, None
        except Exception as e:
            logger.error(f"Exception while fetching {url}: {type(e).__name__}: {e}")
            self.metrics.record_request_result(url, success=False)
            return url, None

    async def async_fetch_urls(self, urls: List[str], batch_size: int = 3, timeout: int = 30) -> Dict[str, Optional[str]]:
        """
        Asynchronously fetch content from multiple URLs in batches.
        
        Args:
            urls: List of URLs to fetch
            batch_size: Maximum number of concurrent requests
            timeout: Maximum time to wait for all requests to complete
            
        Returns:
            Dict[str, Optional[str]]: Dictionary mapping URLs to content (or None if failed)
        """
        # Validate input
        if not urls:
            return {}
            
        # Record start time
        start_time = time.time()
            
        # Process URLs in batches to avoid overwhelming servers
        results = {}
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_start = time.time()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size} with {len(batch)} URLs")
            
            # Create tasks for this batch
            tasks = [asyncio.create_task(self.async_fetch_url(url)) for url in batch]
            
            # Wait for all tasks to complete with timeout
            try:
                batch_results = await asyncio.gather(*tasks)
                
                # Add to results
                for url, content in batch_results:
                    results[url] = content
                    
                batch_duration = time.time() - batch_start
                logger.info(f"Batch completed in {batch_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
        
        # Record total duration
        total_duration = time.time() - start_time
        
        # Log and return results
        success_count = sum(1 for content in results.values() if content is not None)
        logger.info(f"Completed {len(urls)} URL fetches ({success_count} successful) in {total_duration:.2f}s")
        
        return results
    
    def validate_content_quality(self, url: str, content: str) -> Tuple[bool, float, str]:
        """
        Validate the quality of fetched content to filter error pages and low-quality content.
        
        Args:
            url: Source URL
            content: HTML content
            
        Returns:
            Tuple[bool, float, str]: (is_valid, quality_score, reason)
        """
        if not content:
            return False, 0.0, "Empty content"
            
        # Try to parse HTML
        try:
            soup = BeautifulSoup(content, 'lxml')
            text = soup.get_text(separator=' ', strip=True)
            
            # Check for common error page indicators
            error_patterns = [
                "404", "not found", "page not found", 
                "error", "server error", "unavailable", 
                "cannot be found", "no longer exists",
                "access denied", "forbidden", "login required"
            ]
            
            title = soup.title.text.lower() if soup.title else ""
            
            # Check for error indicators in title
            for pattern in error_patterns:
                if pattern in title:
                    return False, 0.0, f"Error page detected: '{pattern}' in title"
            
            # Check content length
            if len(text) < 200:
                return False, 0.1, "Content too short (less than 200 chars)"
                
            # Check for paywall/subscription indicators
            paywall_patterns = [
                "subscribe", "subscription", "sign up", "sign in", "log in",
                "premium content", "premium access", "member", "subscribe now"
            ]
            
            paywall_score = 0
            for pattern in paywall_patterns:
                if pattern in text.lower():
                    paywall_score += 1
                    
            if paywall_score >= 3:
                return False, 0.2, "Possible paywall/subscription content"
                
            # Calculate information density (rough estimate)
            info_density = len(set(text.split())) / max(len(text.split()), 1)
            if info_density < 0.3:
                return False, 0.3, f"Low information density ({info_density:.2f})"
                
            # Calculate quality score (0.0-1.0)
            quality_score = min(1.0, (0.4 + (len(text) / 10000) * 0.3 + info_density * 0.3))
            
            return True, quality_score, "Content validated"
            
        except Exception as e:
            logger.error(f"Content validation error for {url}: {e}")
            return False, 0.0, f"Validation error: {str(e)}"
    
    def search_web_with_async(self, query: str, topic_info: Dict = None, max_results: int = 3) -> List[Dict]:
        """
        Perform a web search using asynchronous requests for better performance.
        
        Args:
            query: User query
            topic_info: Topic classification result or None for auto-detection
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Ranked and processed web search results
        """
        logger.info(f"Performing asynchronous web search for query: '{query}'")
        
        # Determine topic category if not provided
        if not topic_info:
            topic_info = self.get_best_topic_category(query)
            topic_category = topic_info["topic"]
        else:
            topic_category = topic_info["topic"]
        
        # Get approved domains for this category
        approved_urls = APPROVED_DOMAINS.get(topic_category, [])
        
        # For multi-topic search, also consider secondary topics based on confidence
        consider_secondary = (topic_info["confidence"] < 0.7)
        
        if consider_secondary:
            logger.info(f"Low confidence ({topic_info['confidence']:.2f}) in primary topic, adding secondary topics")
            
            # Get additional topic(s) based on the query
            all_scores = {}
            for topic in APPROVED_DOMAINS.keys():
                if topic != topic_category:  # Skip primary topic
                    # Simple keyword matching for secondary topics
                    topic_words = topic.replace("_", " ").lower().split()
                    query_words = query.lower().split()
                    
                    # Check for overlap
                    overlap = any(word in query_words for word in topic_words)
                    if overlap:
                        all_scores[topic] = 0.4  # Default score for matches
                        
            # Add top secondary topic if exists
            if all_scores:
                secondary_topic = max(all_scores.items(), key=lambda x: x[1])[0]
                secondary_urls = APPROVED_DOMAINS.get(secondary_topic, [])
                
                # Combine URLs (2/3 primary, 1/3 secondary)
                primary_count = min(len(approved_urls), int(max_results * 2/3) + 1)
                secondary_count = min(len(secondary_urls), max_results - primary_count)
                
                combined_urls = approved_urls[:primary_count] + secondary_urls[:secondary_count]
                logger.info(f"Added {secondary_count} URLs from secondary topic '{secondary_topic}'")
                approved_urls = combined_urls
        
        if not approved_urls:
            logger.warning(f"No approved URLs found for topic: {topic_category}")
            return []
        
        # Ensure all URLs have schemes
        urls_to_fetch = []
        for url in approved_urls:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            urls_to_fetch.append(url)
        
        logger.info(f"Searching topic '{topic_category}' with {len(urls_to_fetch)} URLs")
        
        # Run async fetching through an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Fetch all URLs asynchronously
            results = loop.run_until_complete(self.async_fetch_urls(urls_to_fetch))
            
            # Close the session if it exists
            if self.session:
                loop.run_until_complete(self.session.close())
                self.session = None
        finally:
            loop.close()
        
        # Process successful results
        all_chunks = []
        for url, html_content in results.items():
            if not html_content:
                continue
            
            # Validate content quality
            is_valid, quality_score, reason = self.validate_content_quality(url, html_content)
            if not is_valid:
                logger.warning(f"Low quality content from {url}: {reason}")
                continue
                
            # Extract text
            text_content = self.extract_text_from_html(html_content)
            if not text_content:
                continue
                
            # Chunk the text
            chunks = self.chunk_text(text_content)
            
            # Add URL and quality info to metadata
            for chunk in chunks:
                chunk["metadata"]["url"] = url
                chunk["metadata"]["topic"] = topic_category
                chunk["metadata"]["quality_score"] = quality_score
            
            all_chunks.extend(chunks)
        
        # Score and rank chunks
        scored_chunks = self.score_web_results(query, all_chunks)
        
        # Return top results
        top_results = scored_chunks[:max_results]
        logger.info(f"Returning {len(top_results)} web search results")
        
        # Save metrics after search
        self.metrics.save_metrics()
        
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
        
        # Get topic classification
        topic_info = self.get_best_topic_category(query)
        print(f"📊 Topic identified: {topic_info['topic']} (confidence: {topic_info['confidence']:.2f})")
        
        # Use query analysis to enhance topic detection if provided
        if query_analysis and 'entities' in query_analysis:
            # Try to match entities to topic categories
            entity_topics = []
            for entity in query_analysis['entities']:
                for topic in APPROVED_DOMAINS.keys():
                    if entity.lower() in topic.replace('_', ' ').lower():
                        entity_topics.append(topic)
            
            # If entities strongly suggest a different topic
            if entity_topics and entity_topics[0] != topic_info['topic'] and topic_info['confidence'] < 0.7:
                print(f"🔄 Adjusting topic based on entity analysis: {entity_topics[0]}")
                topic_info['topic'] = entity_topics[0]
        
        # Perform search with async processing
        search_results = self.search_web_with_async(query, topic_info, max_results)
        
        if not search_results:
            print("❌ No relevant web results found.")
            return []
        
        print(f"✅ Found {len(search_results)} relevant web results.")
        return search_results
    
    def _get_cache_key(self, url: str) -> str:
        """Generate a unique cache key for a URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if a cache entry is still valid (not expired)."""
        # Cache entries valid for 24 hours
        cache_ttl = 86400  # 24 hours in seconds
        
        if 'timestamp' not in cache_entry:
            return False
            
        age = time.time() - cache_entry['timestamp']
        return age < cache_ttl
    
    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content with enhanced processing.
        
        Args:
            html_content: HTML content to process
            
        Returns:
            str: Extracted text content
        """
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove unwanted elements
            for unwanted in soup.find_all(['script', 'style', 'iframe', 'header', 'footer', 'nav', 'aside']):
                unwanted.extract()
            
            # Remove comment elements
            for comment in soup.find_all(string=lambda text: isinstance(text, (Comment))):
                comment.extract()
                
            # Extract article content if available for better signal
            main_content = None
            
            # Try to find main content container
            for container in ['article', 'main', '.content', '#content', '.main-content', '.article-content']:
                if container.startswith('.') or container.startswith('#'):
                    main_content = soup.select_one(container)
                else:
                    main_content = soup.find(container)
                    
                if main_content:
                    break
            
            # If main content found, use it, otherwise use full body
            content_soup = main_content if main_content else soup
            
            # Keep paragraph structure for better context
            paragraphs = []
            for p in content_soup.find_all('p'):
                text = p.get_text(strip=True)
                if text and len(text) > 20:  # Skip very short paragraphs (menu items, etc)
                    paragraphs.append(text)
            
            # Also extract headings to maintain document structure
            for h in content_soup.find_all(['h1', 'h2', 'h3']):
                heading_text = h.get_text(strip=True)
                if heading_text:
                    paragraphs.append(f"\n{heading_text}\n")
            
            # If we didn't find enough paragraphs, fall back to generic text extraction
            if len(paragraphs) < 3:
                text = content_soup.get_text(separator='\n', strip=True)
                
                # Clean up the text
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'\n+', '\n', text)
                
                return text
                
            # Join paragraphs with newlines to maintain structure
            return '\n\n'.join(paragraphs)
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def chunk_text(self, text: str, max_chunk_size: int = 512) -> List[Dict]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List[Dict]: List of text chunks with metadata
        """
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds max size, finalize current chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {}
                })
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk with spacing
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if there's content
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {}
            })
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks.append({
                "text": "",
                "metadata": {}
            })
        
        return chunks
    
    def score_web_results(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Score and rank web search results based on relevance to query.
        
        Args:
            query: User query
            chunks: Text chunks from web search
            
        Returns:
            List[Dict]: Scored and ranked chunks
        """
        if not chunks:
            return []
        
        # Get query embedding for comparison
        query_embedding = np.array(embed_text(query), dtype="float32")
        
        # Score each chunk
        for chunk in chunks:
            # Skip empty chunks
            if not chunk["text"]:
                chunk["confidence"] = 0
                continue
            
            # Create embedding for chunk text
            chunk_embedding = np.array(embed_text(chunk["text"][:1000]), dtype="float32")
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            
            # Adjust similarity based on quality score if available
            quality_boost = chunk["metadata"].get("quality_score", 0.75)
            
            # Calculate final confidence score
            confidence = float(similarity) * quality_boost
            
            # Add word match bonus
            query_words = set(query.lower().split())
            chunk_words = set(chunk["text"].lower().split())
            
            word_match_ratio = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
            word_match_bonus = word_match_ratio * 0.1
            
            # Store the final score as confidence
            chunk["confidence"] = min(0.99, confidence + word_match_bonus)
        
        # Sort by confidence score (descending)
        scored_chunks = sorted(chunks, key=lambda x: x.get("confidence", 0), reverse=True)
        
        return scored_chunks