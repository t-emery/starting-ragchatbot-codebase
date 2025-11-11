"""Test the actual live system to diagnose 'query failed' issue"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from config import config
from rag_system import RAGSystem


class TestLiveSystem:
    """Test actual live RAG system"""

    @pytest.fixture
    def live_system(self):
        """Create real RAG system"""
        return RAGSystem(config)

    def test_simple_greeting(self, live_system):
        """Test simple non-content query"""
        print("\n" + "="*60)
        print("TEST: Simple greeting query")
        print("="*60)

        try:
            response, sources = live_system.query("Hello")

            print(f"✓ Query executed successfully")
            print(f"Response: {response[:200]}...")
            print(f"Sources: {len(sources)}")

            assert isinstance(response, str)
            assert len(response) > 0
            print("✓ Test PASSED")

        except Exception as e:
            print(f"✗ Query FAILED with exception: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Query failed: {e}")

    def test_content_query_about_mcp(self, live_system):
        """Test content query that should use search tool"""
        print("\n" + "="*60)
        print("TEST: Content query about MCP")
        print("="*60)

        try:
            response, sources = live_system.query("What is MCP?")

            print(f"✓ Query executed successfully")
            print(f"Response length: {len(response)} chars")
            print(f"Response preview: {response[:300]}...")
            print(f"Sources: {len(sources)}")

            if sources:
                print("Source details:")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")

            assert isinstance(response, str)
            assert len(response) > 0
            print("✓ Test PASSED")

        except Exception as e:
            print(f"✗ Query FAILED with exception: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Content query failed: {e}")

    def test_content_query_about_chroma(self, live_system):
        """Test content query about Chroma"""
        print("\n" + "="*60)
        print("TEST: Content query about Chroma")
        print("="*60)

        try:
            response, sources = live_system.query("What is Chroma used for?")

            print(f"✓ Query executed successfully")
            print(f"Response length: {len(response)} chars")
            print(f"Response preview: {response[:300]}...")
            print(f"Sources: {len(sources)}")

            if sources:
                print("Source details:")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")

            assert isinstance(response, str)
            assert len(response) > 0
            print("✓ Test PASSED")

        except Exception as e:
            print(f"✗ Query FAILED with exception: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Content query failed: {e}")

    def test_query_with_course_filter(self, live_system):
        """Test query about specific course"""
        print("\n" + "="*60)
        print("TEST: Query about specific lessons in MCP course")
        print("="*60)

        try:
            response, sources = live_system.query("What topics are covered in the MCP course?")

            print(f"✓ Query executed successfully")
            print(f"Response length: {len(response)} chars")
            print(f"Response: {response}")
            print(f"Sources: {len(sources)}")

            if sources:
                print("Source details:")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")

            assert isinstance(response, str)
            assert len(response) > 0
            print("✓ Test PASSED")

        except Exception as e:
            print(f"✗ Query FAILED with exception: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Query with course context failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
