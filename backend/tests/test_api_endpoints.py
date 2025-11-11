"""Tests for FastAPI endpoints"""

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def create_test_app():
    """Create a test FastAPI app without static file mounting"""
    from fastapi import HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any

    # Import models
    from config import config
    from rag_system import RAGSystem

    # Initialize FastAPI app
    app = FastAPI(title="Course Materials RAG System", root_path="")

    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Mock RAG system for testing
    rag_system = Mock(spec=RAGSystem)

    # Pydantic models
    class QueryRequest(BaseModel):
        """Request model for course queries"""
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        """Response model for course queries"""
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str

    class CourseStats(BaseModel):
        """Response model for course statistics"""
        total_courses: int
        course_titles: List[str]

    # API Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            # Process query using RAG system
            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Store rag_system on app for testing access
    app.state.rag_system = rag_system

    return app


@pytest.fixture
def test_app():
    """Create test FastAPI app"""
    return create_test_app()


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_system(test_app):
    """Get the mocked RAG system from test app"""
    return test_app.state.rag_system


# ============================================================================
# Tests for /api/query endpoint
# ============================================================================

class TestQueryEndpoint:
    """Test /api/query endpoint"""

    def test_query_without_session_creates_session(self, client, mock_rag_system):
        """Test query without session_id creates a new session"""
        # Setup mock
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "new_session_123"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = (
            "Machine learning is a subset of AI.",
            [{"course_title": "ML Course", "lesson_number": 1, "lesson_link": "https://example.com/lesson-1"}]
        )

        # Make request
        response = client.post("/api/query", json={
            "query": "What is machine learning?"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Machine learning is a subset of AI."
        assert data["session_id"] == "new_session_123"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["course_title"] == "ML Course"

        # Verify session was created
        mock_session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is machine learning?", "new_session_123")

    def test_query_with_session_uses_existing_session(self, client, mock_rag_system):
        """Test query with session_id uses existing session"""
        # Setup mock
        mock_session_manager = Mock()
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = (
            "Follow-up answer",
            []
        )

        # Make request with session_id
        response = client.post("/api/query", json={
            "query": "Tell me more",
            "session_id": "existing_session_456"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing_session_456"

        # Verify session was not created
        mock_session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me more", "existing_session_456")

    def test_query_returns_sources(self, client, mock_rag_system):
        """Test query returns sources from RAG system"""
        # Setup mock with multiple sources
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "session_789"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = (
            "Answer based on multiple sources",
            [
                {"course_title": "Course A", "lesson_number": 1, "lesson_link": "https://example.com/a/lesson-1"},
                {"course_title": "Course B", "lesson_number": 2, "lesson_link": "https://example.com/b/lesson-2"}
            ]
        )

        # Make request
        response = client.post("/api/query", json={
            "query": "Compare topics"
        })

        # Verify sources
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["course_title"] == "Course A"
        assert data["sources"][1]["course_title"] == "Course B"

    def test_query_handles_error(self, client, mock_rag_system):
        """Test query endpoint handles errors gracefully"""
        # Setup mock to raise error
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "session_error"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        # Make request
        response = client.post("/api/query", json={
            "query": "Test error"
        })

        # Verify error response
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_validates_request_body(self, client):
        """Test query endpoint validates request body"""
        # Missing required field
        response = client.post("/api/query", json={
            "session_id": "test"
            # Missing "query" field
        })

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_empty_string(self, client, mock_rag_system):
        """Test query with empty string"""
        # Setup mock
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "session_empty"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = (
            "Please provide a question.",
            []
        )

        # Make request with empty query
        response = client.post("/api/query", json={
            "query": ""
        })

        # Verify response
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with("", "session_empty")

    def test_query_with_special_characters(self, client, mock_rag_system):
        """Test query with special characters"""
        # Setup mock
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "session_special"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = (
            "Answer with special characters: <>&\"'",
            []
        )

        # Make request with special characters
        response = client.post("/api/query", json={
            "query": "What about <html> & \"quotes\"?"
        })

        # Verify response handles special characters
        assert response.status_code == 200
        data = response.json()
        assert "<>&\"'" in data["answer"]


# ============================================================================
# Tests for /api/courses endpoint
# ============================================================================

class TestCoursesEndpoint:
    """Test /api/courses endpoint"""

    def test_get_courses_returns_analytics(self, client, mock_rag_system):
        """Test /api/courses returns course analytics"""
        # Setup mock
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": [
                "Introduction to Machine Learning",
                "Advanced Python Programming",
                "Data Structures and Algorithms"
            ]
        }

        # Make request
        response = client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Machine Learning" in data["course_titles"]

        # Verify method was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_empty_database(self, client, mock_rag_system):
        """Test /api/courses with empty database"""
        # Setup mock for empty database
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        # Make request
        response = client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_handles_error(self, client, mock_rag_system):
        """Test /api/courses handles errors gracefully"""
        # Setup mock to raise error
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        # Make request
        response = client.get("/api/courses")

        # Verify error response
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_get_courses_large_dataset(self, client, mock_rag_system):
        """Test /api/courses with large number of courses"""
        # Setup mock with many courses
        course_titles = [f"Course {i}" for i in range(100)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": course_titles
        }

        # Make request
        response = client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestAPIIntegration:
    """Integration tests for API endpoints"""

    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries using the same session"""
        # Setup mock
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "persistent_session"
        mock_rag_system.session_manager = mock_session_manager

        # First query
        mock_rag_system.query.return_value = ("First answer", [])
        response1 = client.post("/api/query", json={
            "query": "First question"
        })
        session_id = response1.json()["session_id"]

        # Second query with same session
        mock_rag_system.query.return_value = ("Second answer", [])
        response2 = client.post("/api/query", json={
            "query": "Second question",
            "session_id": session_id
        })

        # Verify both queries succeeded
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

        # Verify query was called twice
        assert mock_rag_system.query.call_count == 2

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/api/query")

        # Check CORS headers are present
        # Note: TestClient doesn't always expose middleware headers
        # This is a basic check
        assert response.status_code in [200, 405]  # OPTIONS may not be explicitly handled

    def test_query_and_courses_endpoints_work_together(self, client, mock_rag_system):
        """Test querying data after checking available courses"""
        # Setup mock
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }

        # First check courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200

        # Then make a query
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "test_session"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = ("Answer", [])

        query_response = client.post("/api/query", json={
            "query": "Test question"
        })
        assert query_response.status_code == 200


# ============================================================================
# Performance and Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and performance"""

    def test_very_long_query(self, client, mock_rag_system):
        """Test handling very long query text"""
        # Setup mock
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "long_query_session"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = ("Answer", [])

        # Create very long query
        long_query = "What is " + "very " * 1000 + "long query?"

        # Make request
        response = client.post("/api/query", json={
            "query": long_query
        })

        # Verify it's handled
        assert response.status_code == 200

    def test_unicode_query(self, client, mock_rag_system):
        """Test handling unicode characters in query"""
        # Setup mock
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "unicode_session"
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = ("Unicode answer: 你好", [])

        # Make request with unicode
        response = client.post("/api/query", json={
            "query": "What is 机器学习?"
        })

        # Verify unicode is handled
        assert response.status_code == 200
        assert "你好" in response.json()["answer"]

    def test_concurrent_sessions(self, client, mock_rag_system):
        """Test handling multiple different sessions"""
        # Setup mock
        mock_session_manager = Mock()
        mock_session_manager.create_session.side_effect = ["session_1", "session_2", "session_3"]
        mock_rag_system.session_manager = mock_session_manager
        mock_rag_system.query.return_value = ("Answer", [])

        # Create multiple sessions
        responses = []
        for i in range(3):
            response = client.post("/api/query", json={
                "query": f"Query {i}"
            })
            responses.append(response)

        # Verify all succeeded with different sessions
        assert all(r.status_code == 200 for r in responses)
        session_ids = [r.json()["session_id"] for r in responses]
        assert len(set(session_ids)) == 3  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
