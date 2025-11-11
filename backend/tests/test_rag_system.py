"""Tests for RAG System end-to-end integration"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from rag_system import RAGSystem


@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def mock_rag_system(mock_config):
    """Create RAGSystem with mocked components"""
    with patch('rag_system.DocumentProcessor'), \
         patch('rag_system.VectorStore'), \
         patch('rag_system.AIGenerator'), \
         patch('rag_system.SessionManager'), \
         patch('rag_system.CourseSearchTool'), \
         patch('rag_system.CourseOutlineTool'):
        system = RAGSystem(mock_config)
        return system


class TestRAGSystemInitialization:
    """Test RAG system initialization"""

    def test_initialization_creates_all_components(self, mock_config):
        """Test that RAGSystem initializes all required components"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr, \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            system = RAGSystem(mock_config)

            # Verify all components initialized
            mock_doc_proc.assert_called_once()
            mock_vector_store.assert_called_once()
            mock_ai_gen.assert_called_once()
            mock_session_mgr.assert_called_once()

            # Verify components exist
            assert hasattr(system, 'document_processor')
            assert hasattr(system, 'vector_store')
            assert hasattr(system, 'ai_generator')
            assert hasattr(system, 'session_manager')
            assert hasattr(system, 'tool_manager')

    def test_tools_registered(self, mock_rag_system):
        """Test that search tools are registered"""
        # Verify tool manager has registered tools
        assert hasattr(mock_rag_system, 'tool_manager')
        assert hasattr(mock_rag_system, 'search_tool')
        assert hasattr(mock_rag_system, 'outline_tool')


class TestRAGSystemQuery:
    """Test RAG system query method"""

    def test_query_without_session(self, mock_rag_system):
        """Test query without session ID"""
        # Mock AI generator response
        mock_rag_system.ai_generator.generate_response = Mock(return_value="This is the answer")
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute query
        response, sources = mock_rag_system.query("What is machine learning?")

        # Verify
        assert response == "This is the answer"
        assert isinstance(sources, list)
        mock_rag_system.ai_generator.generate_response.assert_called_once()

    def test_query_with_session(self, mock_rag_system):
        """Test query with session ID includes conversation history"""
        # Mock components
        mock_rag_system.session_manager.get_conversation_history = Mock(
            return_value="User: Previous question\nAssistant: Previous answer"
        )
        mock_rag_system.ai_generator.generate_response = Mock(return_value="Follow-up answer")
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()
        mock_rag_system.session_manager.add_exchange = Mock()

        # Execute query with session
        session_id = "test_session_123"
        response, sources = mock_rag_system.query("Follow-up question", session_id=session_id)

        # Verify history was retrieved
        mock_rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)

        # Verify history was updated
        mock_rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Follow-up answer"
        )

    def test_query_passes_tools_to_ai(self, mock_rag_system):
        """Test query passes tool definitions to AI generator"""
        # Mock tool definitions
        mock_tools = [
            {"name": "search_course_content", "description": "Search courses"}
        ]
        mock_rag_system.tool_manager.get_tool_definitions = Mock(return_value=mock_tools)
        mock_rag_system.ai_generator.generate_response = Mock(return_value="Answer")
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute query
        response, sources = mock_rag_system.query("test question")

        # Verify tools were passed
        call_args = mock_rag_system.ai_generator.generate_response.call_args
        assert call_args[1]['tools'] == mock_tools
        assert call_args[1]['tool_manager'] == mock_rag_system.tool_manager

    def test_query_returns_sources(self, mock_rag_system):
        """Test query returns sources from tool searches"""
        # Mock sources
        expected_sources = [
            {"course_title": "ML Course", "lesson_number": 1, "lesson_link": "https://example.com/lesson-1"}
        ]
        mock_rag_system.ai_generator.generate_response = Mock(return_value="Answer")
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=expected_sources)
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute query
        response, sources = mock_rag_system.query("test")

        # Verify sources returned
        assert sources == expected_sources

    def test_query_resets_sources_after_retrieval(self, mock_rag_system):
        """Test sources are reset after being retrieved"""
        # Mock components
        mock_rag_system.ai_generator.generate_response = Mock(return_value="Answer")
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute query
        response, sources = mock_rag_system.query("test")

        # Verify reset was called
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_with_content_question(self, mock_rag_system):
        """Test query with content-related question triggers tool use"""
        # Simulate tool execution
        mock_sources = [{"course_title": "ML Course", "lesson_number": 1}]
        mock_rag_system.ai_generator.generate_response = Mock(
            return_value="Machine learning is a subset of AI that learns from data."
        )
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute content query
        response, sources = mock_rag_system.query("What is machine learning?")

        # Verify response and sources
        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(sources, list)


class TestRAGSystemDocumentProcessing:
    """Test document processing methods"""

    def test_add_course_document(self, mock_rag_system):
        """Test adding a single course document"""
        # Mock course and chunks
        mock_course = Mock()
        mock_course.title = "Test Course"
        mock_chunks = [Mock(), Mock(), Mock()]

        mock_rag_system.document_processor.process_course_document = Mock(
            return_value=(mock_course, mock_chunks)
        )
        mock_rag_system.vector_store.add_course_metadata = Mock()
        mock_rag_system.vector_store.add_course_content = Mock()

        # Add document
        course, chunk_count = mock_rag_system.add_course_document("/path/to/course.txt")

        # Verify
        assert course == mock_course
        assert chunk_count == 3
        mock_rag_system.vector_store.add_course_metadata.assert_called_once_with(mock_course)
        mock_rag_system.vector_store.add_course_content.assert_called_once_with(mock_chunks)

    def test_add_course_document_handles_error(self, mock_rag_system):
        """Test error handling when adding document"""
        # Mock error
        mock_rag_system.document_processor.process_course_document = Mock(
            side_effect=Exception("Parse error")
        )

        # Add document
        course, chunk_count = mock_rag_system.add_course_document("/path/to/bad.txt")

        # Verify error handled
        assert course is None
        assert chunk_count == 0

    def test_get_course_analytics(self, mock_rag_system):
        """Test getting course analytics"""
        # Mock analytics
        mock_rag_system.vector_store.get_course_count = Mock(return_value=5)
        mock_rag_system.vector_store.get_existing_course_titles = Mock(
            return_value=["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        )

        # Get analytics
        analytics = mock_rag_system.get_course_analytics()

        # Verify
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5


class TestRAGSystemIntegration:
    """Integration tests for RAG system components"""

    def test_full_query_flow_with_tool_execution(self, mock_rag_system):
        """Test complete query flow including tool execution"""
        # Setup: Mock tool execution scenario
        mock_rag_system.tool_manager.get_tool_definitions = Mock(return_value=[
            {"name": "search_course_content"}
        ])

        # Simulate AI deciding to use tool and getting results
        mock_rag_system.ai_generator.generate_response = Mock(
            return_value="Based on the course content, machine learning is..."
        )

        mock_sources = [
            {
                "course_title": "Introduction to ML",
                "lesson_number": 1,
                "lesson_link": "https://example.com/ml/lesson-1"
            }
        ]
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute query
        response, sources = mock_rag_system.query("What is machine learning?")

        # Verify complete flow
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(sources) > 0
        assert sources[0]["course_title"] == "Introduction to ML"
        assert sources[0]["lesson_number"] == 1

        # Verify all methods called
        mock_rag_system.ai_generator.generate_response.assert_called_once()
        mock_rag_system.tool_manager.get_last_sources.assert_called_once()
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_without_tool_use(self, mock_rag_system):
        """Test query that doesn't require tool use"""
        # Setup: General knowledge question
        mock_rag_system.ai_generator.generate_response = Mock(
            return_value="Hello! I'm an AI assistant that helps with course materials."
        )
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute general query
        response, sources = mock_rag_system.query("Hello, who are you?")

        # Verify response without sources
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(sources) == 0

    def test_session_continuity(self, mock_rag_system):
        """Test conversation continuity across multiple queries"""
        session_id = "test_session_456"

        # Mock session manager
        mock_rag_system.session_manager.get_conversation_history = Mock(
            side_effect=[
                None,  # First query - no history
                "User: What is ML?\nAssistant: ML is machine learning.",  # Second query - has history
            ]
        )
        mock_rag_system.session_manager.add_exchange = Mock()
        mock_rag_system.ai_generator.generate_response = Mock(
            side_effect=[
                "ML is machine learning.",
                "It's used for pattern recognition and predictions."
            ]
        )
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # First query
        response1, _ = mock_rag_system.query("What is ML?", session_id=session_id)

        # Second query with context
        response2, _ = mock_rag_system.query("What is it used for?", session_id=session_id)

        # Verify history was used in second query
        assert mock_rag_system.session_manager.get_conversation_history.call_count == 2
        assert mock_rag_system.session_manager.add_exchange.call_count == 2


class TestErrorScenarios:
    """Test error handling in various scenarios"""

    def test_query_with_ai_error(self, mock_rag_system):
        """Test query handles AI generation error"""
        # Mock AI error
        mock_rag_system.ai_generator.generate_response = Mock(
            side_effect=Exception("API Error")
        )

        # Execute query - should raise exception
        with pytest.raises(Exception) as exc_info:
            mock_rag_system.query("test")

        assert "API Error" in str(exc_info.value)

    def test_query_with_empty_database(self, mock_rag_system):
        """Test query behavior with empty database"""
        # Simulate empty search results
        mock_rag_system.ai_generator.generate_response = Mock(
            return_value="No relevant content found in the database."
        )
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Execute query
        response, sources = mock_rag_system.query("What is ML?")

        # Verify graceful handling
        assert isinstance(response, str)
        assert len(sources) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
