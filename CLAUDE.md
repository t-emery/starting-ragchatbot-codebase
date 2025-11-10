# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot** for querying course materials. It uses a two-tier architecture:
- **Backend**: Python FastAPI server with ChromaDB vector database
- **Frontend**: Vanilla HTML/CSS/JavaScript web interface
- **AI**: Anthropic Claude with tool-calling for autonomous search

## Running the Application

### Start Server
```bash
./run.sh
# OR
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
Create `.env` in project root:
```
ANTHROPIC_API_KEY=your_key_here
```

### Install Dependencies
```bash
uv sync
```

### Access Points
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture

### Request Flow (2-Turn Tool Calling Pattern)

```
User Query → FastAPI → RAGSystem → AIGenerator → Claude API (Turn 1)
                                                      ↓ (decides to search)
                                                   tool_use
                                                      ↓
                                    ToolManager → CourseSearchTool
                                                      ↓
                                    VectorStore → ChromaDB (semantic search)
                                                      ↓
                                    Results → Claude API (Turn 2)
                                                      ↓
                                    Final Answer → User
```

**Key Pattern**: Claude makes two API calls per query:
1. **Turn 1**: Receives user question + tool definitions, decides whether to search
2. **Turn 2**: Receives search results, synthesizes final answer

### Core Components

**RAGSystem** (`backend/rag_system.py`)
- Main orchestrator coordinating all components
- Entry point: `query(query: str, session_id: str) -> (response, sources)`
- Manages tool execution and conversation history

**AIGenerator** (`backend/ai_generator.py`)
- Handles Claude API interactions with tool calling
- `generate_response()`: Makes initial API call
- `_handle_tool_execution()`: Executes tools and makes follow-up call
- System prompt defines search behavior and response format

**VectorStore** (`backend/vector_store.py`)
- ChromaDB wrapper with two collections:
  - `course_catalog`: Course metadata (title, instructor, lessons)
  - `course_content`: Searchable text chunks with embeddings
- `search()`: Unified search interface with course name resolution and content search
- Uses `all-MiniLM-L6-v2` SentenceTransformer for embeddings

**DocumentProcessor** (`backend/document_processor.py`)
- Parses structured course documents (expected format in comments)
- `chunk_text()`: Sentence-based chunking with overlap (800 chars, 100 char overlap)
- `process_course_document()`: Extracts course metadata and creates chunks with context prefixes

**ToolManager + CourseSearchTool** (`backend/search_tools.py`)
- Implements Claude tool-calling interface
- `CourseSearchTool.execute()`: Main search logic
- `_format_results()`: Formats chunks for Claude and tracks sources for UI
- Sources tracked in `last_sources` attribute for UI attribution

**SessionManager** (`backend/session_manager.py`)
- Maintains conversation history per session
- `MAX_HISTORY` limit (default: 2 exchanges = 4 messages)
- History injected into Claude's system prompt for context

### Data Models (`backend/models.py`)

- **Course**: title (unique ID), instructor, lessons[], course_link
- **Lesson**: lesson_number, title, lesson_link
- **CourseChunk**: content, course_title, lesson_number, chunk_index

### Configuration (`backend/config.py`)

Key settings loaded from `config` singleton:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800 chars
- `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 chunks per search
- `MAX_HISTORY`: 2 exchanges
- `CHROMA_PATH`: "./chroma_db"

## Document Processing

### Expected Document Format
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: Introduction
Lesson Link: [url]
[lesson content...]

Lesson 1: Next Topic
[lesson content...]
```

### Adding Documents
Place `.txt`, `.pdf`, or `.docx` files in `/docs` folder. They're auto-loaded on startup via `app.py:startup_event()`.

### Chunking Strategy
- Splits on sentence boundaries (regex handles abbreviations)
- Builds chunks up to `CHUNK_SIZE` without breaking sentences
- Overlaps `CHUNK_OVERLAP` characters between chunks
- First chunk of each lesson prefixed with: `"Lesson {N} content: {text}"`
- Last lesson chunks prefixed with: `"Course {title} Lesson {N} content: {text}"`

## Vector Search Mechanics

**Course Name Resolution**:
- When user provides partial course name (e.g., "Python intro")
- System does semantic search on `course_catalog` collection
- Returns exact course title for filtering

**Content Search**:
- Query embedded using SentenceTransformer
- ChromaDB performs cosine similarity search
- Optional filters: `course_title` and/or `lesson_number`
- Returns top `MAX_RESULTS` chunks with metadata

**Source Attribution**:
- Search tool tracks sources in `last_sources` list
- Format: `"{course_title} - Lesson {lesson_number}"`
- Retrieved via `ToolManager.get_last_sources()` after response
- Displayed in UI as collapsible "Sources" section

## API Endpoints

**POST /api/query**
- Request: `{query: str, session_id?: str}`
- Response: `{answer: str, sources: str[], session_id: str}`

**GET /api/courses**
- Response: `{total_courses: int, course_titles: str[]}`

## Modifying AI Behavior

**System Prompt** (`ai_generator.py:SYSTEM_PROMPT`):
- Defines when to search vs. use general knowledge
- "One search per query maximum" constraint
- Response formatting rules (concise, no meta-commentary)

**Tool Definition** (`search_tools.py:CourseSearchTool.get_tool_definition()`):
- Tool name: `search_course_content`
- Parameters: `query` (required), `course_name`, `lesson_number`
- Description guides Claude on when/how to use tool

## Frontend-Backend Contract

**JavaScript (`frontend/script.js`)**:
- Sends query + session_id to `/api/query`
- Receives answer + sources
- Uses `marked.parse()` to render markdown responses
- Displays sources in collapsible details element

**Session Persistence**:
- First query: `session_id: null` → backend creates new session
- Subsequent queries: reuse `session_id` from previous response
- Session history maintained server-side in `SessionManager`

## Database Persistence

ChromaDB persists to `./chroma_db/` directory:
- `course_catalog` collection: Course metadata
- `course_content` collection: Text chunks + embeddings

**Clearing Data**:
```python
rag_system.add_course_folder(docs_path, clear_existing=True)
```

**Deduplication**:
- Uses course title as unique ID
- `add_course_folder()` skips courses already in `course_catalog`

## Key Implementation Patterns

**Tool Execution Flow**:
1. Claude's response has `stop_reason="tool_use"`
2. Extract `content_block.name` and `content_block.input`
3. Execute via `ToolManager.execute_tool(name, **input)`
4. Build messages array: `[user, assistant (tool_use), user (tool_results)]`
5. Second API call **without tools** parameter to get final answer

**Error Handling**:
- Empty search results: Tool returns "No relevant content found"
- Invalid course name: VectorStore returns `SearchResults.empty(error_msg)`
- Tool errors: Returned as string to Claude for handling

**Context Management**:
- Session history formatted as: `"User: {msg}\nAssistant: {msg}\n..."`
- Appended to system prompt before each request
- History truncated to last `MAX_HISTORY * 2` messages (user+assistant pairs)
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run Python files