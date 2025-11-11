"""Diagnostic tests to check actual system state and identify issues

These tests check the REAL system state (not mocked) to diagnose issues:
- Is ChromaDB populated with courses?
- Is the API key configured?
- Can we connect to the database?
- Can we perform basic operations?

Run these tests to identify the root cause of "query failed" errors.
"""

import os
import sys
from pathlib import Path

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import after adding to path
from config import config
from rag_system import RAGSystem
from vector_store import VectorStore


class TestSystemConfiguration:
    """Test system configuration and environment"""

    def test_anthropic_api_key_exists(self):
        """CRITICAL: Test that Anthropic API key is configured"""
        api_key = config.ANTHROPIC_API_KEY

        # Check if key exists and is not empty
        assert api_key is not None, "ANTHROPIC_API_KEY is None - not loaded from environment"
        assert api_key != "", "ANTHROPIC_API_KEY is empty string - check .env file"
        assert len(api_key) > 10, f"ANTHROPIC_API_KEY seems too short: {len(api_key)} chars"

        print(f"✓ API key configured (length: {len(api_key)} chars)")

    def test_config_values_loaded(self):
        """Test that all required config values are loaded"""
        assert hasattr(config, "ANTHROPIC_MODEL"), "ANTHROPIC_MODEL not in config"
        assert hasattr(config, "EMBEDDING_MODEL"), "EMBEDDING_MODEL not in config"
        assert hasattr(config, "CHUNK_SIZE"), "CHUNK_SIZE not in config"
        assert hasattr(config, "MAX_RESULTS"), "MAX_RESULTS not in config"
        assert hasattr(config, "CHROMA_PATH"), "CHROMA_PATH not in config"

        print(f"✓ Configuration loaded:")
        print(f"  - Model: {config.ANTHROPIC_MODEL}")
        print(f"  - Embedding: {config.EMBEDDING_MODEL}")
        print(f"  - ChromaDB path: {config.CHROMA_PATH}")

    def test_chroma_path_exists(self):
        """Test that ChromaDB directory exists"""
        chroma_path = Path(backend_path) / config.CHROMA_PATH

        if chroma_path.exists():
            print(f"✓ ChromaDB directory exists: {chroma_path}")
        else:
            print(f"✗ ChromaDB directory NOT FOUND: {chroma_path}")
            print("  This might be okay if no documents have been loaded yet")


class TestDatabaseState:
    """Test actual database state"""

    @pytest.fixture
    def vector_store(self):
        """Create real VectorStore instance"""
        chroma_path = os.path.join(backend_path, config.CHROMA_PATH)
        return VectorStore(
            chroma_path=chroma_path,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )

    def test_database_has_courses(self, vector_store):
        """CRITICAL: Test if database is populated with courses"""
        course_count = vector_store.get_course_count()

        if course_count == 0:
            print("✗ DATABASE IS EMPTY - No courses found!")
            print("  This is likely the root cause of 'query failed' errors")
            print("  Solution: Load course documents into the database")
            pytest.fail("No courses in database - cannot answer content queries")
        else:
            print(f"✓ Database has {course_count} courses")

    def test_database_has_content(self, vector_store):
        """Test if database has content chunks"""
        try:
            # Try to get some content
            results = vector_store.course_content.get(limit=1)

            if results and "ids" in results and len(results["ids"]) > 0:
                print(f"✓ Database has content chunks")
            else:
                print("✗ Database has NO content chunks")
                pytest.fail("No content chunks in database")
        except Exception as e:
            print(f"✗ Error accessing content: {e}")
            pytest.fail(f"Cannot access course content: {e}")

    def test_list_available_courses(self, vector_store):
        """List all courses in the database for diagnosis"""
        try:
            course_titles = vector_store.get_existing_course_titles()

            print(f"\n📚 Courses in database ({len(course_titles)}):")
            for title in course_titles:
                print(f"  - {title}")

            if len(course_titles) == 0:
                print("  (No courses loaded)")
        except Exception as e:
            print(f"✗ Error listing courses: {e}")

    def test_sample_search_works(self, vector_store):
        """Test if basic search operation works"""
        try:
            # Perform a basic search
            results = vector_store.search(query="test")

            if results.error:
                print(f"✗ Search returned error: {results.error}")
                pytest.fail(f"Search failed: {results.error}")
            else:
                print(f"✓ Search operation works")
                print(f"  Results found: {len(results.documents)}")
        except Exception as e:
            print(f"✗ Search threw exception: {e}")
            pytest.fail(f"Search failed with exception: {e}")


class TestDocumentLoading:
    """Test document loading functionality"""

    def test_docs_folder_exists(self):
        """Test if docs folder exists"""
        # Try common locations
        possible_paths = [
            Path(backend_path).parent / "docs",
            Path(backend_path) / "docs",
            Path(backend_path) / "../docs",
        ]

        found = False
        for path in possible_paths:
            if path.exists():
                print(f"✓ Docs folder found: {path.resolve()}")
                files = list(path.glob("*"))
                print(f"  Files in docs: {len(files)}")
                for f in files:
                    if f.suffix.lower() in [".txt", ".pdf", ".docx"]:
                        print(f"    - {f.name}")
                found = True
                break

        if not found:
            print("✗ Docs folder not found in expected locations:")
            for path in possible_paths:
                print(f"  - Checked: {path.resolve()}")
            print("  Documents may not have been loaded")


class TestRAGSystemIntegration:
    """Test RAG system can be initialized"""

    def test_rag_system_initializes(self):
        """Test that RAG system can be initialized"""
        try:
            system = RAGSystem(config)
            print("✓ RAG system initialized successfully")

            # Check components
            assert system.vector_store is not None
            assert system.ai_generator is not None
            assert system.tool_manager is not None
            print("✓ All RAG components initialized")
        except Exception as e:
            print(f"✗ RAG system failed to initialize: {e}")
            pytest.fail(f"Cannot initialize RAG system: {e}")

    def test_rag_system_has_tools(self):
        """Test that tools are registered"""
        try:
            system = RAGSystem(config)
            tool_defs = system.tool_manager.get_tool_definitions()

            print(f"✓ Tool manager has {len(tool_defs)} tools registered:")
            for tool_def in tool_defs:
                print(f"  - {tool_def['name']}")

            assert len(tool_defs) > 0, "No tools registered"
        except Exception as e:
            print(f"✗ Error checking tools: {e}")
            pytest.fail(f"Tool check failed: {e}")


class TestEndToEndQuery:
    """Test end-to-end query with REAL system (may fail if API key invalid)"""

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_real_query_execution(self):
        """Test a real query end-to-end (skipped by default)"""
        try:
            system = RAGSystem(config)

            # Try a simple query
            response, sources = system.query("Hello")

            print(f"✓ Query executed successfully")
            print(f"  Response length: {len(response)} chars")
            print(f"  Sources: {len(sources)}")

            assert isinstance(response, str)
            assert len(response) > 0
        except Exception as e:
            print(f"✗ Query failed: {e}")
            pytest.fail(f"Real query execution failed: {e}")


class TestDiagnosticSummary:
    """Summary diagnostic test"""

    def test_system_health_check(self):
        """Comprehensive system health check"""
        issues = []
        warnings = []

        # Check 1: API Key
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "":
            issues.append("ANTHROPIC_API_KEY not configured")
        else:
            print("✓ API key configured")

        # Check 2: Database
        try:
            chroma_path = os.path.join(backend_path, config.CHROMA_PATH)
            vector_store = VectorStore(chroma_path, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            course_count = vector_store.get_course_count()

            if course_count == 0:
                issues.append("Database is empty - no courses loaded")
            else:
                print(f"✓ Database has {course_count} courses")
        except Exception as e:
            issues.append(f"Cannot access database: {e}")

        # Check 3: RAG System
        try:
            system = RAGSystem(config)
            print("✓ RAG system initializes")
        except Exception as e:
            issues.append(f"RAG system initialization failed: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        if issues:
            print("\n🔴 CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        if warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        if not issues and not warnings:
            print("\n✅ All checks passed!")

        print("\n" + "=" * 60)

        # Fail test if critical issues found
        if issues:
            pytest.fail(f"Found {len(issues)} critical issue(s)")


if __name__ == "__main__":
    # Run diagnostics with verbose output
    pytest.main([__file__, "-v", "-s"])
