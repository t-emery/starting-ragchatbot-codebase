"""Pytest fixtures and test data for RAG chatbot tests"""

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults

# ============================================================================
# FastAPI Testing Fixtures
# ============================================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client

    Note: Import test_api_endpoints.create_test_app to use this fixture
    for API endpoint testing
    """
    from fastapi.testclient import TestClient
    from test_api_endpoints import create_test_app

    app = create_test_app()
    return TestClient(app)


@pytest.fixture
def mock_fastapi_request():
    """Mock FastAPI request object"""
    mock_request = Mock()
    mock_request.query = "What is machine learning?"
    mock_request.session_id = None
    return mock_request


# ============================================================================
# Sample Test Data
# ============================================================================


@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "title": "Introduction to Machine Learning",
        "instructor": "Dr. Jane Smith",
        "course_link": "https://example.com/ml-course",
        "lessons": [
            {
                "lesson_number": 1,
                "lesson_title": "What is Machine Learning?",
                "lesson_link": "https://example.com/ml-course/lesson-1",
            },
            {
                "lesson_number": 2,
                "lesson_title": "Supervised Learning Basics",
                "lesson_link": "https://example.com/ml-course/lesson-2",
            },
        ],
    }


@pytest.fixture
def sample_chunks():
    """Sample course chunks for testing"""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "course_title": "Introduction to Machine Learning",
            "lesson_number": 1,
            "chunk_index": 0,
        },
        {
            "content": "Supervised learning uses labeled data to train models. Common algorithms include linear regression and decision trees.",
            "course_title": "Introduction to Machine Learning",
            "lesson_number": 2,
            "chunk_index": 0,
        },
    ]


@pytest.fixture
def sample_search_results(sample_chunks):
    """Sample SearchResults for testing"""
    documents = [chunk["content"] for chunk in sample_chunks]
    metadata = [
        {
            "course_title": chunk["course_title"],
            "lesson_number": chunk["lesson_number"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in sample_chunks
    ]
    distances = [0.1, 0.2]

    return SearchResults(documents=documents, metadata=metadata, distances=distances)


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for testing"""
    return SearchResults.empty("No results found")


# ============================================================================
# Mock VectorStore
# ============================================================================


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Mock VectorStore for testing"""
    mock_store = Mock()
    mock_store.search.return_value = sample_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/ml-course/lesson-1"
    mock_store._resolve_course_name.return_value = "Introduction to Machine Learning"
    mock_store.max_results = 5
    return mock_store


@pytest.fixture
def mock_empty_vector_store(empty_search_results):
    """Mock VectorStore that returns empty results"""
    mock_store = Mock()
    mock_store.search.return_value = empty_search_results
    mock_store.get_lesson_link.return_value = None
    mock_store._resolve_course_name.return_value = None
    mock_store.max_results = 5
    return mock_store


# ============================================================================
# Mock ChromaDB Collections
# ============================================================================


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection"""
    mock_collection = Mock()
    mock_collection.query.return_value = {
        "documents": [["Sample document content"]],
        "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
        "distances": [[0.1]],
        "ids": [["doc1"]],
    }
    mock_collection.get.return_value = {
        "ids": ["Test Course"],
        "metadatas": [
            {
                "title": "Test Course",
                "course_link": "https://example.com/test",
                "instructor": "Test Instructor",
                "lessons_json": '[{"lesson_number": 1, "lesson_title": "Lesson 1"}]',
            }
        ],
        "documents": ["Course metadata"],
    }
    return mock_collection


@pytest.fixture
def mock_empty_chroma_collection():
    """Mock ChromaDB collection with no data"""
    mock_collection = Mock()
    mock_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "ids": [[]],
    }
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    return mock_collection


# ============================================================================
# Mock Anthropic Client
# ============================================================================


@pytest.fixture
def mock_anthropic_response_no_tool():
    """Mock Anthropic API response without tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_content = Mock()
    mock_content.text = "This is a general response without using tools."
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_anthropic_response_with_tool():
    """Mock Anthropic API response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create tool use content block
    mock_tool_use = Mock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.id = "tool_123"
    mock_tool_use.name = "search_course_content"
    mock_tool_use.input = {
        "query": "What is machine learning?",
        "course_name": None,
        "lesson_number": None,
    }

    mock_response.content = [mock_tool_use]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Mock Anthropic API final response after tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_content = Mock()
    mock_content.text = (
        "Machine learning is a subset of AI that enables systems to learn from data."
    )
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response_no_tool):
    """Mock Anthropic client"""
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_anthropic_response_no_tool
    return mock_client


# ============================================================================
# Tool Manager Fixtures
# ============================================================================


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"},
                },
                "required": ["query"],
            },
        }
    ]
    mock_manager.execute_tool.return_value = "Search results here"
    mock_manager.get_last_sources.return_value = [
        {
            "course_title": "Introduction to Machine Learning",
            "lesson_number": 1,
            "lesson_link": "https://example.com/ml-course/lesson-1",
        }
    ]
    mock_manager.reset_sources.return_value = None
    return mock_manager


# ============================================================================
# Session Manager Fixtures
# ============================================================================


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = (
        "User: Previous question\nAssistant: Previous answer"
    )
    mock_manager.add_exchange.return_value = None
    return mock_manager


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration object with common test settings"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


# ============================================================================
# API Response Fixtures
# ============================================================================

@pytest.fixture
def sample_query_response():
    """Sample API query response"""
    return {
        "answer": "Machine learning is a subset of artificial intelligence.",
        "sources": [
            {
                "course_title": "Introduction to Machine Learning",
                "lesson_number": 1,
                "lesson_link": "https://example.com/ml-course/lesson-1"
            }
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_courses_response():
    """Sample API courses response"""
    return {
        "total_courses": 3,
        "course_titles": [
            "Introduction to Machine Learning",
            "Advanced Python Programming",
            "Data Structures and Algorithms"
        ]
    }


# ============================================================================
# RAG System Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system_full():
    """Fully configured mock RAG system for API testing"""
    mock_system = Mock()

    # Mock session manager
    mock_session_mgr = Mock()
    mock_session_mgr.create_session.return_value = "test_session_123"
    mock_session_mgr.get_conversation_history.return_value = None
    mock_session_mgr.add_exchange.return_value = None
    mock_system.session_manager = mock_session_mgr

    # Mock query method
    mock_system.query.return_value = (
        "Machine learning is a subset of AI.",
        [{"course_title": "ML Course", "lesson_number": 1, "lesson_link": "https://example.com/lesson-1"}]
    )

    # Mock analytics method
    mock_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course 1", "Course 2", "Course 3"]
    }

    return mock_system
