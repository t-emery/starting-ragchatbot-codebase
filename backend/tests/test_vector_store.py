"""Tests for VectorStore search functionality and database state"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestSearchResults:
    """Test SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [
                [
                    {"course_title": "Test", "lesson_number": 1},
                    {"course_title": "Test", "lesson_number": 2},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        assert results.error is None
        assert not results.is_empty()

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 0
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "Course not found"
        results = SearchResults.empty(error_msg)

        assert results.is_empty()
        assert results.error == error_msg
        assert len(results.documents) == 0


class TestVectorStoreSearch:
    """Test VectorStore search functionality"""

    @patch("vector_store.chromadb.PersistentClient")
    def test_search_without_filters(self, mock_client_class):
        """Test basic search without course or lesson filters"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Machine learning content"]],
            "metadatas": [[{"course_title": "ML Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection

        # Create VectorStore
        store = VectorStore(
            chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        # Execute search
        results = store.search(query="What is machine learning?")

        # Verify
        assert not results.is_empty()
        assert len(results.documents) == 1
        assert "Machine learning content" in results.documents[0]
        mock_collection.query.assert_called_once()

    @patch("vector_store.chromadb.PersistentClient")
    def test_search_with_course_filter(self, mock_client_class):
        """Test search with course name filter"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = {
            "documents": [["ML Course"]],
            "metadatas": [[{"title": "Machine Learning Course"}]],
        }

        mock_content = MagicMock()
        mock_content.query.return_value = {
            "documents": [["Course content"]],
            "metadatas": [[{"course_title": "Machine Learning Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        # Return catalog first, then content
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Execute search with course name
        results = store.search(query="What is ML?", course_name="ML Course")

        # Verify course resolution was called
        mock_catalog.query.assert_called_once()
        # Verify content search was called with filter
        mock_content.query.assert_called_once()
        call_args = mock_content.query.call_args
        assert call_args[1]["where"] == {"course_title": "Machine Learning Course"}

    @patch("vector_store.chromadb.PersistentClient")
    def test_search_with_invalid_course(self, mock_client_class):
        """Test search with non-existent course name"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}

        mock_content = MagicMock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Execute search with invalid course
        results = store.search(query="test", course_name="NonExistentCourse")

        # Verify error is returned
        assert results.is_empty()
        assert results.error is not None
        assert "No course found matching" in results.error

    @patch("vector_store.chromadb.PersistentClient")
    def test_search_with_lesson_filter(self, mock_client_class):
        """Test search with lesson number filter"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_content = MagicMock()
        mock_content.query.return_value = {
            "documents": [["Lesson 1 content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Execute search with lesson filter
        results = store.search(query="test", lesson_number=1)

        # Verify filter was applied
        call_args = mock_content.query.call_args
        assert call_args[1]["where"] == {"lesson_number": 1}

    @patch("vector_store.chromadb.PersistentClient")
    def test_search_with_both_filters(self, mock_client_class):
        """Test search with both course and lesson filters"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = {
            "documents": [["ML Course"]],
            "metadatas": [[{"title": "ML Course"}]],
        }

        mock_content = MagicMock()
        mock_content.query.return_value = {
            "documents": [["Content"]],
            "metadatas": [[{"course_title": "ML Course", "lesson_number": 2}]],
            "distances": [[0.1]],
        }

        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Execute search
        results = store.search(query="test", course_name="ML", lesson_number=2)

        # Verify combined filter
        call_args = mock_content.query.call_args
        assert "$and" in call_args[1]["where"]

    @patch("vector_store.chromadb.PersistentClient")
    def test_search_with_exception(self, mock_client_class):
        """Test search handles ChromaDB exceptions"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_content = MagicMock()
        mock_content.query.side_effect = Exception("ChromaDB error")

        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Execute search
        results = store.search(query="test")

        # Verify error is captured
        assert results.is_empty()
        assert results.error is not None
        assert "Search error" in results.error


class TestVectorStoreDatabaseState:
    """Test VectorStore database state and initialization"""

    @patch("vector_store.chromadb.PersistentClient")
    def test_get_course_count_with_data(self, mock_client_class):
        """Test getting course count when courses exist"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_catalog.get.return_value = {"ids": ["Course1", "Course2", "Course3"]}

        mock_content = MagicMock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Get count
        count = store.get_course_count()

        assert count == 3

    @patch("vector_store.chromadb.PersistentClient")
    def test_get_course_count_empty(self, mock_client_class):
        """Test getting course count when no courses exist"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_catalog.get.return_value = {"ids": []}

        mock_content = MagicMock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Get count
        count = store.get_course_count()

        assert count == 0

    @patch("vector_store.chromadb.PersistentClient")
    def test_get_existing_course_titles(self, mock_client_class):
        """Test retrieving all course titles"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_catalog = MagicMock()
        mock_catalog.get.return_value = {"ids": ["ML Course", "Python Course"]}

        mock_content = MagicMock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        # Get titles
        titles = store.get_existing_course_titles()

        assert len(titles) == 2
        assert "ML Course" in titles
        assert "Python Course" in titles


class TestBuildFilter:
    """Test filter building logic"""

    @patch("vector_store.chromadb.PersistentClient")
    def test_no_filters(self, mock_client_class):
        """Test filter when no parameters provided"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        filter_dict = store._build_filter(None, None)
        assert filter_dict is None

    @patch("vector_store.chromadb.PersistentClient")
    def test_course_filter_only(self, mock_client_class):
        """Test filter with only course title"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        filter_dict = store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    @patch("vector_store.chromadb.PersistentClient")
    def test_lesson_filter_only(self, mock_client_class):
        """Test filter with only lesson number"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        filter_dict = store._build_filter(None, 5)
        assert filter_dict == {"lesson_number": 5}

    @patch("vector_store.chromadb.PersistentClient")
    def test_both_filters(self, mock_client_class):
        """Test filter with both course and lesson"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")

        filter_dict = store._build_filter("Test Course", 3)
        assert "$and" in filter_dict
        assert {"course_title": "Test Course"} in filter_dict["$and"]
        assert {"lesson_number": 3} in filter_dict["$and"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
