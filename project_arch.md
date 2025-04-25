# System Analysis Report: Histronaut History Tutor

## Introduction

### Purpose of the System
The Histronaut (also referred to as Yuhasa in the codebase) is a specialized Retrieval-Augmented Generation (RAG) chatbot designed to function as a virtual history tutor. The primary goal of the system is to provide students with an interactive learning experience by answering questions about Grade 11 history content, pulling information directly from a history textbook. Unlike conventional AI chatbots that rely solely on pre-trained knowledge, this system grounds its responses in the specific content of the textbook, reducing hallucination and providing more accurate, curriculum-aligned answers.

### Core Technologies Used
- **Python**: Core programming language (v3.10+ recommended)
- **Flask**: Web framework for the user interface and API endpoints
- **Google Gemini**: Large language model for text generation and reasoning
- **FAISS (Facebook AI Similarity Search)**: Vector database for efficient similarity searching
- **PyPDF/PyMuPDF**: Libraries for processing and chunking the PDF textbook
- **Sentence Transformers/Embedding Models**: For converting text into vector representations
- **HTML/CSS/JavaScript**: Front-end interface components

### Intended Audience and Use Case
The system is primarily designed for:
- **Grade 11 Students**: Seeking help understanding history concepts from their textbook
- **Teachers**: As a supplementary resource to support student learning
- **Educational Institutions**: Looking to implement AI-assisted learning tools

The primary use case involves students asking questions about historical events, figures, or concepts covered in their Grade 11 history textbook and receiving accurate, contextual explanations from the AI tutor.

## System Architecture Overview

### High-level Diagram of Components
```
┌─────────────────────────────┐      ┌─────────────────────────────┐
│       User Interfaces       │      │     Data Preprocessing      │
│                             │      │                             │
│  ┌─────────┐  ┌─────────┐  │      │  ┌─────────┐  ┌─────────┐  │
│  │   Web   │  │  Command│  │      │  │   PDF   │  │  FAISS  │  │
│  │Interface│  │  Line   │  │      │  │Chunking │  │  Index  │  │
│  └─────────┘  └─────────┘  │      │  └─────────┘  └─────────┘  │
└─────────────────────────────┘      └─────────────────────────────┘
           │                                       │
           ▼                                       ▼
┌─────────────────────────────┐      ┌─────────────────────────────┐
│       RAG Pipeline          │      │      Storage Systems        │
│                             │      │                             │
│  ┌─────────┐  ┌─────────┐  │      │  ┌─────────┐  ┌─────────┐  │
│  │ Query   │  │Retriever│  │◄────►│  │Vector DB│  │Metadata │  │
│  │Analyzer │  │Agent    │  │      │  │(FAISS)  │  │Storage  │  │
│  └─────────┘  └─────────┘  │      │  └─────────┘  └─────────┘  │
│        │           │       │      └─────────────────────────────┘
│        ▼           ▼       │                    │
│  ┌─────────┐  ┌─────────┐  │                    │
│  │Context  │  │Generator│  │                    │
│  │Expander │  │Agent    │  │                    │
│  └─────────┘  └─────────┘  │                    │
│        │           │       │                    │
│        └────┬──────┘       │                    │
│             ▼              │                    │
│  ┌────────────────────┐    │                    │
│  │  Orchestrator      │◄───┘                    │
│  │  Agent             │                         │
│  └────────────────────┘                         │
└─────────────────────────────┘                   │
           │                                      │
           ▼                                      │
┌─────────────────────────────┐                   │
│    Session Management       │                   │
│                             │                   │
│  ┌─────────┐  ┌─────────┐  │                   │
│  │  Chat   │  │User Data│  │◄──────────────────┘
│  │History  │  │Storage  │  │
│  └─────────┘  └─────────┘  │
└─────────────────────────────┘
```

### Data Flow from User Query to Response
1. **User Input**: The user submits a question via the web interface or CLI
2. **Query Analysis**: The query is analyzed to extract keywords, entities, and query type
3. **Retrieval**: The system searches for relevant chunks from the textbook using semantic search
4. **Context Expansion**: Additional chunks may be retrieved to expand the context
5. **Answer Generation**: The LLM (Gemini) generates an answer using the retrieved context
6. **Response Delivery**: The formatted answer is presented to the user
7. **Session Management**: The interaction is saved to the user's chat history

### Key Directories and Files Structure
```
/
├── agents/                    # Core RAG pipeline components
│   ├── __init__.py
│   ├── base.py                # Base agent class
│   ├── context_expander.py    # Expands retrieved context
│   ├── generator.py           # Generates answers using Gemini
│   ├── orchestrator.py        # Coordinates the entire pipeline
│   ├── query_analyzer.py      # Analyzes user queries
│   ├── reference_tracker.py   # Tracks source references
│   └── retriever.py           # Retrieves relevant chunks
├── chats/                     # Stores user chat histories
├── static/                    # Static assets for web UI
├── templates/                 # HTML templates
│   └── index.html             # Main web interface
├── app.py                     # Alternative Flask application entry
├── cli_chat.py                # Command-line interface
├── config.py                  # Configuration management
├── embed_store.py             # Creates embeddings
├── faiss_store.py             # FAISS indexing utilities
├── gemini_utils.py            # Google Gemini API utilities
├── main.py                    # FastAPI version of the server
├── pdf_chunker.py             # PDF processing and chunking
├── web.py                     # Primary Flask application
├── faiss_index.index          # Vector index (binary)
└── faiss_metadata.pkl         # Chunk metadata (binary)
```

## Data Ingestion and Processing

### PDF Chunking Strategy and Implementation
The system processes the Grade 11 history textbook PDF through the following steps:
1. **Document Loading**: The PDF is loaded using PyMuPDF (via the `fitz` module)
2. **Text Extraction**: Text is extracted page by page, preserving page numbers
3. **Chunking Strategy**: The text is divided into semantically meaningful chunks (paragraphs or sections) with appropriate overlap to avoid losing context at chunk boundaries
4. **Metadata Preservation**: Each chunk is associated with metadata including page number, section title (if available), and position in the document

The chunking approach balances granularity with context preservation:
- Chunks are sized to capture complete thoughts or sections (typically 200-500 tokens)
- Consecutive chunks have overlapping content to maintain continuity
- Section titles and page numbers are preserved for citation and reference

### Text Embedding Process and Model Choice
The system converts text chunks into vector embeddings using the following process:
1. **Text Normalization**: Chunks are normalized (lowercase, whitespace normalization)
2. **Embedding Generation**: The `embed_text` function in `gemini_utils.py` generates embeddings
3. **Dimensionality**: The embeddings are high-dimensional vectors (typically 384 or 768 dimensions depending on the model)

The embedding model used appears to be from Google's API, though specific model details are not explicitly stated in the codebase. The embedding process captures semantic meaning, enabling similarity-based search beyond simple keyword matching.

### FAISS Vector Store Creation and Configuration
The FAISS index is created and configured through these steps:
1. **Vector Collection**: All chunk embeddings are collected into a single numpy array
2. **Index Construction**: A FAISS IndexFlatL2 index is created (L2 distance for similarity)
3. **Vector Storage**: The vectors are added to the index
4. **Persistence**: The index is saved to disk as `faiss_index.index`

Configuration details:
- The system uses a flat index for maximum accuracy (at the cost of some performance)
- L2 (Euclidean) distance is used for similarity measurement
- The index is optimized for search speed over memory efficiency

### Metadata Extraction and Storage
Alongside vector embeddings, the system maintains rich metadata:
1. **Metadata Capture**: For each chunk, metadata is extracted including:
   - Page number
   - Section title (when available)
   - Paragraph position
   - Source information
2. **Storage Format**: Metadata is stored in a Python dictionary and serialized using pickle
3. **Persistence**: The metadata is saved as `faiss_metadata.pkl`

The metadata structure links each vector in the FAISS index back to its source text and context information, enabling proper citation and context-aware presentation of information.

## User Interfaces

### Web Interface (Flask, HTML/CSS/JS)
The web interface provides an intuitive chatbot experience:
1. **Key Features**:
   - Responsive design that works on both desktop and mobile
   - Dark/light theme toggle
   - Chat history sidebar with session management
   - Markdown rendering for formatted responses
   - File attachment capability (though not fully implemented for document upload)
   - Quick suggestion buttons for common queries

2. **Technical Implementation**:
   - Built with vanilla HTML/CSS/JavaScript
   - Uses Bootstrap for basic styling and FontAwesome for icons
   - Client-side persistence using localStorage for chat sessions
   - AJAX for asynchronous communication with the backend

3. **Design Approach**:
   - Clean, modern interface with a focus on readability
   - Glass-morphism visual style with responsive layout
   - Mobile-friendly design with collapsible sidebar
   - Bot personality expressed through avatar and response styling

### API Endpoints and Their Functions
The system exposes several API endpoints:

1. **`/` (GET)**: 
   - Renders the main chat interface
   - Initializes user session if not present
   - Loads existing chat history

2. **`/ask` (POST)**:
   - Primary endpoint for question answering
   - Accepts: `query` (user question) and `chat_history` (previous messages)
   - Returns: Generated answer and updated chat history
   - Handles: Context retrieval, answer generation, and history management

3. **Potential Additional Endpoints** (not fully implemented):
   - `/upload`: For document upload capabilities
   - `/feedback`: For collecting user feedback on responses
   - `/export`: For exporting chat transcripts

### Chat History Management
The system implements a comprehensive chat history management system:

1. **Session Management**:
   - Each user receives a unique session ID stored in Flask's session
   - Sessions persist between visits via browser cookies

2. **Storage Strategy**:
   - Chat histories are stored in JSON format in the `chats/` directory
   - Each user's history is in a separate file named with their user ID
   - Multiple chat sessions per user are supported

3. **Features**:
   - Creation of new chat sessions
   - Renaming and deletion of sessions
   - Automatic titling based on first user message
   - Periodic summary generation to manage context window size

4. **Client-Server Synchronization**:
   - Browser localStorage maintains chat history on the client
   - Server storage ensures persistence across devices
   - Merge strategy handles conflicts when present

## RAG Pipeline Components

### OrchestratorAgent
**Purpose and Responsibilities**:
- Acts as the central coordinator for the entire RAG pipeline
- Manages communication between all component agents
- Directs the flow of data through the system
- Assembles the final response from component outputs

**Input/Output Specifications**:
- **Inputs**:
  - User query (string)
  - Chat history (optional list)
- **Outputs**:
  - Complete answer object containing:
    - Generated answer text
    - Source references
    - Query analysis details
    - Retrieved context chunks

**Key Algorithms or Techniques**:
- Sequential pipeline execution with conditional branching
- Error handling and fallback strategies
- Coordination of parallel processes when applicable

**Potential Issues or Limitations**:
- Single point of failure in the system architecture
- Limited fault tolerance if sub-agents fail
- No dynamic allocation of computational resources
- Does not adapt pipeline steps based on query complexity

### QueryAnalyzerAgent
**Purpose and Responsibilities**:
- Analyzes and enhances user queries
- Extracts entities and keywords
- Determines query type and complexity
- Refines queries for improved retrieval

**Input/Output Specifications**:
- **Inputs**:
  - Raw user query (string)
- **Outputs**:
  - Analysis object containing:
    - Refined query
    - Extracted entities and keywords
    - Query type classification
    - Decomposition into sub-queries (for complex questions)

**Key Algorithms or Techniques**:
- Regular expression pattern matching for entity extraction
- Keyword identification using frequency analysis
- Query type classification (factual, analytical, comparative, etc.)
- Query refinement through stopword removal and expansion

**Potential Issues or Limitations**:
- Relies primarily on pattern matching rather than deep NLP
- Limited handling of ambiguous questions
- No spelling correction or advanced query reformulation
- Entity extraction may miss contextual entities

### RetrieverAgent
**Purpose and Responsibilities**:
- Retrieves relevant text chunks from the FAISS index
- Re-ranks results using multiple scoring mechanisms
- Filters irrelevant content
- Assigns confidence scores to retrieved chunks

**Input/Output Specifications**:
- **Inputs**:
  - Query (string)
  - Query analysis (from QueryAnalyzerAgent)
  - Optional parameters (top_k, filters)
- **Outputs**:
  - Ranked list of relevant text chunks with:
    - Text content
    - Metadata (page, section)
    - Relevance scores
    - Confidence metrics

**Key Algorithms or Techniques**:
- Vector similarity search using FAISS
- Hybrid retrieval combining:
  - Semantic similarity (vector distance)
  - BM25-inspired keyword matching
  - Entity match scoring
  - Section relevance heuristics
- Multi-factor re-ranking algorithm with weighted scoring

**Potential Issues or Limitations**:
- Fixed weighting for different scoring factors
- No user feedback loop for retrieval improvement
- Limited context window consideration
- No dynamic adaptation of retrieval strategy based on query type

### ContextExpansionAgent
**Purpose and Responsibilities**:
- Expands retrieved context when necessary
- Ensures context coherence and completeness
- Aggregates related chunks from different sections
- Optimizes context window utilization

**Input/Output Specifications**:
- **Inputs**:
  - Initial retrieved chunks
  - Query analysis
  - Reference to retriever agent (for additional retrievals)
- **Outputs**:
  - Expanded and optimized context chunks
  - Aggregated metadata for references

**Key Algorithms or Techniques**:
- Context window management
- Similarity-based chunk clustering
- Page proximity analysis
- Section continuity detection

**Potential Issues or Limitations**:
- May add irrelevant context in expansion
- Fixed expansion strategies not adapting to query needs
- Limited handling of contradictory information
- No prioritization for most information-dense chunks

### GeneratorAgent
**Purpose and Responsibilities**:
- Generates coherent, accurate answers using LLM
- Formulates effective prompts based on query type
- Ensures answers are grounded in retrieved context
- Applies appropriate formatting and structure

**Input/Output Specifications**:
- **Inputs**:
  - Original query
  - Context chunks
  - Query analysis
  - Optional chat history
- **Outputs**:
  - Formatted answer text with markdown styling

**Key Algorithms or Techniques**:
- Prompt engineering tailored to query types
- Context integration into structured prompts
- Response formatting with markdown
- Citation and reference incorporation

**Potential Issues or Limitations**:
- Reliance on fixed prompt templates
- Limited adaptability to unusual query types
- No multi-step reasoning for complex questions
- Potential for model hallucination despite context

### ReferenceTrackerAgent
**Purpose and Responsibilities**:
- Tracks source references for generated answers
- Maps answer statements to source chunks
- Formats citations in user-friendly way
- Aggregates reference metadata

**Input/Output Specifications**:
- **Inputs**:
  - Generated answer
  - Context chunks with metadata
- **Outputs**:
  - Structured reference object with:
    - Page citations
    - Section references

**Key Algorithms or Techniques**:
- Text alignment between answer and sources
- Metadata aggregation and deduplication
- Citation formatting according to style guidelines

**Potential Issues or Limitations**:
- Limited precision in source attribution
- No explicit citation markers in responses
- Inability to attribute synthesized information
- No verification of citation accuracy

## Data Flow

### Step-by-step Trace of a Query Through the System

1. **User Input Phase**:
   - User types question in web interface or CLI
   - Question is sent to server via AJAX POST to `/ask` endpoint
   - Request includes current chat history

2. **Query Analysis Phase**:
   - `QueryAnalyzerAgent` processes raw query
   - Identifies key entities (e.g., "World War II", "Industrial Revolution")
   - Extracts important keywords
   - Classifies query type (factual, analytical, etc.)
   - Refines query for retrieval

3. **Retrieval Phase**:
   - `RetrieverAgent` converts query to vector embedding
   - FAISS index is searched for similar vectors
   - Initial candidates are retrieved based on vector similarity
   - Candidates are re-ranked using multiple scoring factors:
     - Semantic relevance (vector similarity)
     - Keyword matching (BM25-inspired)
     - Entity presence
     - Section relevance
   - Top candidates are selected based on combined scores

4. **Context Expansion Phase**:
   - `ContextExpansionAgent` examines initial chunks
   - Determines if context needs expansion
   - May request additional chunks from `RetrieverAgent`
   - Aggregates and organizes chunks for coherent context
   - Prepares metadata for references

5. **Generation Phase**:
   - `GeneratorAgent` constructs a prompt including:
     - Original user query
     - Retrieved context chunks
     - Query type-specific instructions
     - Chat history for conversation continuity
   - Prompt is sent to Gemini API
   - Response is received and processed
   - Formatting is applied (markdown, structure)

6. **Post-processing Phase**:
   - `ReferenceTrackerAgent` associates response with sources
   - Formats references for presentation
   - Final answer is assembled with metadata

7. **Response Delivery Phase**:
   - Complete answer is returned to user interface
   - Chat history is updated and stored
   - UI displays response with appropriate formatting

### Data Transformations at Each Stage

1. **Query Transformation**:
   - Raw text → Analyzed query object
   - Text → Vector embedding
   - Query → Enhanced query with metadata

2. **Retrieval Transformations**:
   - Vector → Candidate chunk list
   - Candidates → Scored and ranked chunks
   - Chunks → Filtered relevant context

3. **Context Transformations**:
   - Initial chunks → Expanded context
   - Multiple chunks → Coherent narrative context
   - Source metadata → Aggregated references

4. **Generation Transformations**:
   - Context + Query → Structured prompt
   - Prompt → Raw LLM response
   - Response → Formatted markdown answer

5. **Output Transformations**:
   - Answer + Metadata → Complete response object
   - Response → UI rendering
   - Interaction → Persistent chat history

### Key Decision Points in the Pipeline

1. **Query Complexity Assessment**:
   - Simple vs. complex query handling
   - Query decomposition for multi-part questions

2. **Retrieval Strategy Selection**:
   - Number of chunks to retrieve (top_k)
   - Balance between semantic and keyword matching

3. **Context Expansion Decisions**:
   - Whether to expand context
   - Which additional chunks to include
   - How to handle context window limitations

4. **Prompt Construction Choices**:
   - Prompt template selection based on query type
   - Inclusion of chat history
   - Level of instruction detail

5. **Response Generation Parameters**:
   - Temperature/creativity settings
   - Response formatting requirements
   - Citation inclusion strategy

## Dependencies

### Complete List of External Libraries
- **Core Dependencies**:
  - `flask`: Web framework
  - `google-generativeai`: Google Gemini API interface
  - `faiss-cpu`: Vector similarity search
  - `numpy`: Numerical operations
  - `pypdf`/`PyMuPDF (fitz)`: PDF processing

- **Front-end Dependencies**:
  - `bootstrap`: CSS framework (CDN)
  - `font-awesome`: Icons (CDN)
  - `marked`: Markdown rendering (CDN)

- **Additional Libraries**:
  - `python-dotenv`: Environment variable management
  - `pickle`: Serialization
  - `json`: Data formatting
  - `re`: Regular expressions
  - `collections`: Data structures (Counter)

### Version Requirements and Potential Conflicts
- **Python**: Version 3.10+ recommended
- **FAISS**: Version compatibility issues between CPU and GPU versions
- **PyMuPDF**: Version-specific PDF handling behavior
- **google-generativeai**: Rapidly evolving API may cause breaking changes

### Missing or Implicit Dependencies
- No explicit environment isolation (virtualenv)
- Implicit dependency on browser localStorage for client-side persistence
- No specified CORS handling for potential cross-origin requests
- Missing explicit error logging framework

## Potential Issues and Sources of Inaccuracy

### Chunking and Embedding Quality
- **Chunk Size Trade-offs**: Too small chunks lose context, too large chunks dilute relevance
- **Overlap Strategy**: Insufficient overlap may lose cross-chunk context
- **Embedding Model Limitations**: Semantic nuances may be lost in vector representation
- **Out-of-Vocabulary Terms**: Domain-specific historical terms may not embed well

### Retrieval Effectiveness
- **Vector Search Limitations**: Semantic similarity does not always equate to answer relevance
- **Fixed Re-ranking Weights**: Non-adaptive weighting of different retrieval factors
- **Context Window Constraints**: Limited number of chunks can be included in context
- **Missing Information**: Relevant content may be in the textbook but not retrieved

### Prompt Engineering Challenges
- **Template Rigidity**: Fixed prompt templates may not adapt to all query variations
- **Context Integration**: Inefficient use of context window with repetitive information
- **Instruction Clarity**: Ambiguous instructions may lead to inconsistent responses
- **Chat History Management**: Long conversations may lose important earlier context

### Model Limitations
- **Hallucination Risk**: LLM may generate plausible but incorrect information
- **Reasoning Ability**: Limited multi-step reasoning for complex historical analysis
- **Source Attribution**: Difficulty in precise attribution of information to sources
- **Synthetic Information**: Model may blend information from multiple sources incorrectly

### System Complexity Issues
- **Error Propagation**: Errors in early pipeline stages compound in later stages
- **Session Management**: Potential for session conflicts or data loss
- **Scaling Challenges**: Limited consideration for multi-user load
- **Dependency Fragility**: Complex dependency chain increases failure points

## Recommendations

### Specific Improvements for Each Component
1. **PDF Processing and Chunking**:
   - Implement semantic chunking based on section boundaries
   - Add hierarchical chunking with multiple granularity levels
   - Preserve more structural metadata (headers, lists, tables)

2. **Retrieval System**:
   - Implement query expansion using historical lexicons
   - Add dynamic weighting of retrieval factors based on query type
   - Incorporate user feedback signals into retrieval ranking

3. **Context Management**:
   - Implement semantic deduplication of similar chunks
   - Add dynamic context prioritization based on information density
   - Develop smarter context window management with hierarchical compression

4. **Answer Generation**:
   - Create more specialized prompts for different historical periods
   - Implement chain-of-thought reasoning for analytical questions
   - Add explicit citation markers in responses

5. **User Interface**:
   - Implement full document upload capabilities
   - Add visual timeline for historical events discussed
   - Create topic visualization of chat history

### Performance Optimization Opportunities
1. **Vector Search Optimization**:
   - Implement HNSW or IVF index types for faster searching
   - Add caching layer for common queries
   - Use quantization to reduce memory footprint

2. **Request Processing**:
   - Implement asynchronous processing for non-blocking operations
   - Add request queuing for high-load scenarios
   - Optimize embedding generation with batching

3. **Front-end Improvements**:
   - Implement progressive loading for chat history
   - Add client-side caching of responses
   - Optimize rendering of markdown content

### Accuracy Enhancement Strategies
1. **Retrieval Improvements**:
   - Implement Retrieval-Focused Fine-Tuning (ReFT)
   - Add dense passage retrieval with learned representations
   - Implement hybrid search with structured metadata filters

2. **LLM Enhancements**:
   - Add fact-checking against retrieved context
   - Implement self-consistency checks for answers
   - Add explicit uncertainty indicators for low-confidence responses

3. **Knowledge Augmentation**:
   - Create structured knowledge graph for key historical entities
   - Implement timeline-based reasoning for chronological questions
   - Add support for multimedia content (maps, diagrams)

### Code Refactoring Suggestions
1. **Architecture Improvements**:
   - Implement dependency injection for better modularity
   - Add comprehensive logging framework
   - Create formal API documentation

2. **Testing Infrastructure**:
   - Add unit tests for each component
   - Implement integration tests for the full pipeline
   - Create evaluation benchmarks for retrieval quality

3. **Deployment Enhancements**:
   - Containerize application with Docker
   - Add configurable environment variables
   - Implement monitoring and observability tools

## Conclusion

### Overall Assessment of System Design
The Histronaut History Tutor demonstrates a well-conceived RAG architecture tailored for educational applications. Its component-based design allows for modularity and potential expansion, while the focus on accurate information retrieval from a specific textbook provides a solid foundation for educational use. The system successfully combines modern AI techniques with traditional information retrieval methods to create a specialized educational tool.

### Key Strengths
1. **Domain Specialization**: Focused specifically on history education rather than general-purpose QA
2. **Hybrid Retrieval**: Combines semantic search with keyword matching and metadata awareness
3. **User Experience**: Clean, intuitive interface with conversation management
4. **Context Awareness**: Maintains conversation history for coherent interactions
5. **Educational Design**: Explicitly designed for learning rather than just information retrieval

### Key Weaknesses
1. **Limited Reasoning**: Lacks advanced multi-step reasoning for complex historical analysis
2. **Fixed Strategies**: Non-adaptive pipeline with limited flexibility for different query types
3. **Attribution Precision**: Imprecise source attribution and citation
4. **Scaling Limitations**: Not optimized for multi-user educational environments
5. **Evaluation Framework**: Lacks systematic accuracy evaluation methods

### Priority Areas for Improvement
1. **Retrieval Quality**: Enhancing the accuracy and relevance of retrieved chunks
2. **Reasoning Capabilities**: Adding structured reasoning for analytical history questions
3. **Educational Features**: Developing more learning-focused features like quizzes and explanations
4. **Evaluation Framework**: Creating benchmarks specific to history education
5. **Documentation and Maintenance**: Improving code documentation and maintainability

The Histronaut History Tutor represents a promising application of RAG technology to educational contexts, demonstrating how AI can be effectively constrained to specific knowledge domains for improved accuracy and educational utility. With targeted improvements to its retrieval and reasoning capabilities, it could become an even more valuable tool for history education.