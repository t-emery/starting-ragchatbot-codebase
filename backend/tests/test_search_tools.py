"""Tests for CourseSearchTool.execute() method"""

import pytest
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test CourseSearchTool.execute() method"""

    def test_execute_with_valid_results(self, mock_vector_store, sample_search_results):
        """Test execute returns formatted results when search succeeds"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        result = tool.execute(query="What is machine learning?")

        # Verify
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Machine learning" in result or "Introduction to Machine Learning" in result
        mock_vector_store.search.assert_called_once_with(
            query="What is machine learning?",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test execute passes course_name filter to vector store"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        result = tool.execute(query="What is ML?", course_name="Machine Learning")

        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="What is ML?",
            course_name="Machine Learning",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test execute passes lesson_number filter to vector store"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        result = tool.execute(query="test", lesson_number=2)

        # Verify search was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=2
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test execute passes both course and lesson filters"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        result = tool.execute(
            query="supervised learning",
            course_name="ML Course",
            lesson_number=2
        )

        # Verify both filters passed
        mock_vector_store.search.assert_called_once_with(
            query="supervised learning",
            course_name="ML Course",
            lesson_number=2
        )

    def test_execute_with_empty_results(self, mock_empty_vector_store):
        """Test execute handles empty search results"""
        # Setup
        tool = CourseSearchTool(mock_empty_vector_store)

        # Execute
        result = tool.execute(query="nonexistent topic")

        # Verify error message returned
        assert isinstance(result, str)
        assert "No relevant content found" in result

    def test_execute_with_course_filter_no_results(self, mock_empty_vector_store):
        """Test execute returns appropriate message when course filter yields no results"""
        # Setup
        tool = CourseSearchTool(mock_empty_vector_store)

        # Execute
        result = tool.execute(query="test", course_name="NonExistent Course")

        # Verify message includes course name
        assert "No relevant content found" in result
        assert "NonExistent Course" in result or "course" in result.lower()

    def test_execute_with_error_in_search(self, mock_vector_store):
        """Test execute handles search errors"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = SearchResults.empty("Search error: Database connection failed")

        # Execute
        result = tool.execute(query="test")

        # Verify error propagated
        assert isinstance(result, str)
        assert "Search error" in result or "Database connection failed" in result

    def test_execute_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test execute stores sources in last_sources"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        result = tool.execute(query="test")

        # Verify sources tracked
        assert hasattr(tool, 'last_sources')
        assert len(tool.last_sources) == len(sample_search_results.documents)

    def test_format_results_includes_course_title(self, mock_vector_store, sample_search_results):
        """Test formatted results include course title"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        result = tool.execute(query="test")

        # Verify course title in output
        assert "Introduction to Machine Learning" in result

    def test_format_results_includes_lesson_number(self, mock_vector_store, sample_search_results):
        """Test formatted results include lesson number"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        result = tool.execute(query="test")

        # Verify lesson number in output
        assert "Lesson 1" in result or "Lesson 2" in result

    def test_sources_include_lesson_links(self, mock_vector_store, sample_search_results):
        """Test that sources include lesson links when available"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        # Execute
        result = tool.execute(query="test")

        # Verify sources have lesson links
        assert len(tool.last_sources) > 0
        # At least one source should have a lesson_link
        has_link = any("lesson_link" in source for source in tool.last_sources)
        assert has_link


class TestCourseOutlineTool:
    """Test CourseOutlineTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition is correctly formatted"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert "course_name" in definition["input_schema"]["properties"]

    def test_execute_with_valid_course(self, mock_vector_store):
        """Test execute returns course outline"""
        tool = CourseOutlineTool(mock_vector_store)

        # Mock course catalog response
        mock_vector_store.course_catalog.get.return_value = {
            'ids': ['ML Course'],
            'metadatas': [{
                'title': 'ML Course',
                'instructor': 'Dr. Smith',
                'course_link': 'https://example.com/ml',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Intro"}]'
            }]
        }
        mock_vector_store._resolve_course_name.return_value = "ML Course"

        # Execute
        result = tool.execute(course_name="ML")

        # Verify
        assert "ML Course" in result
        assert "Dr. Smith" in result
        assert "Lesson 1" in result

    def test_execute_with_invalid_course(self, mock_vector_store):
        """Test execute handles invalid course name"""
        tool = CourseOutlineTool(mock_vector_store)
        mock_vector_store._resolve_course_name.return_value = None

        # Execute
        result = tool.execute(course_name="NonExistent")

        # Verify error message
        assert "not found" in result


class TestToolManager:
    """Test ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test executing a tool by name"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = sample_search_results

        # Execute tool
        result = manager.execute_tool("search_course_content", query="test")

        # Verify
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = sample_search_results

        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = manager.get_last_sources()

        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = sample_search_results

        # Execute search
        manager.execute_tool("search_course_content", query="test")

        # Reset sources
        manager.reset_sources()

        # Verify sources cleared
        sources = manager.get_last_sources()
        assert len(sources) == 0


class TestToolDefinitions:
    """Test tool definitions match Anthropic spec"""

    def test_search_tool_definition_structure(self, mock_vector_store):
        """Test CourseSearchTool definition has required fields"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        # Required fields
        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition

        # Input schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Query is required
        assert "query" in schema["required"]

    def test_search_tool_parameters(self, mock_vector_store):
        """Test CourseSearchTool has correct parameters"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        properties = definition["input_schema"]["properties"]

        # Check all expected parameters
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        # Check types
        assert properties["query"]["type"] == "string"
        assert properties["course_name"]["type"] == "string"
        assert properties["lesson_number"]["type"] == "integer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
