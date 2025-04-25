# Project State: Histronaut History Tutor (Web Search Enhancement)

## Current Implementation Status

The Histronaut History Tutor has been enhanced with a domain-restricted web search capability to supplement the existing textbook-based RAG system. This feature allows the system to retrieve information from competition-approved websites only, maintaining strict source control while expanding the knowledge base.

## Key Components Implemented

### 1. Domain Restriction Framework
- Created `config/approved_domains.py` containing all competition-approved domains organized by topic categories
- Implemented domain validation to ensure searches never go outside approved boundaries
- Added topic categorization to map queries to the most relevant domain categories

### 2. WebSearchAgent
- Implemented a dedicated agent for web search within approved domains only
- Created robust caching mechanism to avoid redundant web requests
- Added text extraction from HTML with proper chunking strategy aligned with textbook chunking
- Implemented scoring and ranking for web search results

### 3. RAG Pipeline Integration
- Updated OrchestratorAgent to intelligently combine textbook and web sources
- Enhanced ContextExpansionAgent to handle different source types appropriately
- Modified GeneratorAgent to properly cite and distinguish between textbook and web sources
- Added source attribution and reference tracking for web content

### 4. Security and Error Handling
- Implemented strict URL validation to prevent accidental requests to non-approved sites
- Added comprehensive error handling for network issues and failed requests
- Created fallback strategies for when web search fails or finds no relevant results

## Usage Flow

1. User submits a query to the history tutor
2. QueryAnalyzerAgent analyzes the query for entities, keywords, and query type
3. RetrieverAgent searches for relevant textbook content
4. Based on query content and textbook results, OrchestratorAgent decides if web search is needed
5. If needed, WebSearchAgent retrieves information from approved websites for the relevant topic
6. Retrieved content from both sources is combined and processed by ContextExpansionAgent
7. GeneratorAgent creates a comprehensive answer with proper citation of all sources
8. Results are presented to the user with clear source attribution

## Next Steps

- Enhance topic mapping accuracy for better domain selection
- Implement more advanced ranking for combined textbook and web sources
- Add evaluation metrics to compare answers with and without web search capability
- Create a visualization interface to show source distribution in responses

## Dependencies Added

- requests: For fetching web content
- beautifulsoup4: For HTML parsing and content extraction
- lxml: For efficient HTML parsing
- urllib3: For URL handling and validation
- cachetools: For efficient caching of web search results