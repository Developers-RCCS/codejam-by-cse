# Project State: Histronaut History Tutor (Web Search Optimization)

**Overall Goal:** Enhance the existing RAG chatbot with optimized web search capabilities that leverage advanced query classification, parallel processing, and intelligent source balancing to provide more accurate, faster responses while maintaining strict adherence to approved domains.

**Current System Status:**
- Domain-restricted web search capability successfully implemented
- WebSearchAgent handles fetching and processing from approved domains only
- OrchestratorAgent intelligently decides when to use web search
- Basic topic categorization and result ranking implemented
- GeneratorAgent properly handles citation and source attribution
- Current limitations: sequential processing, basic topic classification, no metrics tracking

**Current Task (Web Search Optimization):**
- Implement embedding-based topic classification for more accurate domain selection
- Add asynchronous web request processing for parallel fetching and faster responses
- Create intelligent source balancing to better handle textbook vs. web content conflicts
- Implement comprehensive metrics tracking to measure web search effectiveness
- Add multi-topic search capability for complex queries spanning multiple domains
- Enhance content validation to filter out low-quality web content

**Next Steps:** After optimizing web search, enhance the frontend to better visualize source attribution and implement advanced RAG techniques.

**Key Technologies:** Python, aiohttp (for async requests), scikit-learn/numpy, asyncio, time/timeit

## Implementation Details

### 1. Enhanced Topic Classification
- Embedding-based similarity between query and topic descriptions
- Weighted combination with keyword-based approach
- Confidence scores included in classification results

### 2. Asynchronous Web Processing
- Parallel URL fetching with aiohttp
- Batched processing to manage server resources
- Timeout handling and retry mechanisms
- Significant response time improvements

### 3. Intelligent Source Balancing
- Source conflict detection for contradictory information
- Reliability scoring system for different sources
- Conflict resolution strategy in OrchestratorAgent
- Multi-perspective presentation for controversial topics

### 4. Comprehensive Metrics System
- Performance tracking (response times, cache hit rates)
- Topic classification accuracy monitoring
- Source distribution analysis
- Web search effectiveness measurement

### 5. Multi-Topic Search Capability
- Cross-topic search for complex queries
- Intelligent distribution of search quota
- Results merging and deduplication

### 6. Enhanced Content Validation
- Quality assessment for web content
- Error page detection
- Content relevance scoring
- Low-information content filtering

## Dependencies Added
- aiohttp: For asynchronous HTTP requests
- scikit-learn: For enhanced similarity calculations
- timeit/time: For performance measurements