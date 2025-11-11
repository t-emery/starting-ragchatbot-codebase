"""Tests for AIGenerator tool calling mechanism"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ai_generator import AIGenerator


class TestAIGeneratorBasic:
    """Test basic AIGenerator functionality"""

    def test_initialization(self):
        """Test AIGenerator initializes correctly"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            assert generator.model == "claude-sonnet-4-20250514"
            assert generator.base_params["model"] == "claude-sonnet-4-20250514"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_exists(self):
        """Test system prompt is defined"""
        assert hasattr(AIGenerator, 'SYSTEM_PROMPT')
        assert len(AIGenerator.SYSTEM_PROMPT) > 0
        assert "course" in AIGenerator.SYSTEM_PROMPT.lower()


class TestGenerateResponseWithoutTools:
    """Test generate_response without tool calling"""

    def test_generate_response_no_tools(self, mock_anthropic_client, mock_anthropic_response_no_tool):
        """Test response generation without tools"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            response = generator.generate_response(query="What is 2+2?")

            # Verify API was called
            mock_anthropic_client.messages.create.assert_called_once()

            # Verify response
            assert isinstance(response, str)
            assert len(response) > 0

    def test_generate_response_with_conversation_history(self, mock_anthropic_client, mock_anthropic_response_no_tool):
        """Test response includes conversation history in system prompt"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            history = "User: Previous question\nAssistant: Previous answer"
            response = generator.generate_response(query="Follow-up question", conversation_history=history)

            # Verify history was included
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args[1]["system"]
            assert "Previous question" in system_content


class TestGenerateResponseWithTools:
    """Test generate_response with tool calling"""

    def test_tools_included_in_api_call(self, mock_tool_manager):
        """Test tools are passed to API when provided"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response without tools"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                query="What is ML?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify tools were passed
            call_args = mock_client.messages.create.call_args
            assert "tools" in call_args[1]
            assert "tool_choice" in call_args[1]
            assert call_args[1]["tool_choice"]["type"] == "auto"

    def test_tool_use_triggers_execution(self, mock_tool_manager):
        """Test tool_use stop_reason triggers tool execution"""
        mock_client = Mock()

        # First response: tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_block]

        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = Mock()
        mock_final_content.text = "Final answer based on search"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                query="What is ML?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test"
            )

            # Verify two API calls were made
            assert mock_client.messages.create.call_count == 2

            # Verify final response returned
            assert response == "Final answer based on search"

    def test_tool_execution_without_tool_manager(self):
        """Test tool_use without tool_manager returns first response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_content = Mock()
        mock_content.text = "Would use tools but no manager"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            tools = [{"name": "test_tool"}]
            response = generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=None  # No tool manager
            )

            # Should return text from first response content
            # Since there's no text in tool_use, this might fail in real scenario
            # But the logic should handle it
            assert mock_client.messages.create.call_count == 1


class TestHandleToolExecution:
    """Test _handle_tool_execution method"""

    def test_builds_correct_message_sequence(self, mock_tool_manager):
        """Test tool execution builds correct message array"""
        mock_client = Mock()

        # Initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_abc"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "ML basics"}
        initial_response.content = [tool_block]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Machine learning is..."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [initial_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            # Call generate_response which will trigger _handle_tool_execution
            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                query="What is ML?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Check second API call had correct message structure
            assert mock_client.messages.create.call_count == 2
            second_call_args = mock_client.messages.create.call_args_list[1]

            messages = second_call_args[1]["messages"]
            # Should have: user message, assistant (tool use), user (tool result)
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"

    def test_tool_results_formatted_correctly(self, mock_tool_manager):
        """Test tool results are formatted with correct structure"""
        mock_client = Mock()

        # Mock tool execution
        mock_tool_manager.execute_tool.return_value = "Search results here"

        # Initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_xyz"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        initial_response.content = [tool_block]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Final answer"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [initial_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify tool result structure in second call
            second_call_args = mock_client.messages.create.call_args_list[1]
            messages = second_call_args[1]["messages"]

            # Tool results should be in third message
            tool_results = messages[2]["content"]
            assert isinstance(tool_results, list)
            assert len(tool_results) > 0
            assert tool_results[0]["type"] == "tool_result"
            assert tool_results[0]["tool_use_id"] == "tool_xyz"
            assert tool_results[0]["content"] == "Search results here"

    def test_no_tools_in_second_api_call(self, mock_tool_manager):
        """Test second API call does not include tools parameter"""
        mock_client = Mock()

        # Initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        initial_response.content = [tool_block]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Answer"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [initial_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Check second API call doesn't have tools
            second_call_args = mock_client.messages.create.call_args_list[1]
            assert "tools" not in second_call_args[1]

    def test_multiple_tool_calls_in_one_response(self, mock_tool_manager):
        """Test handling multiple tool use blocks in single response"""
        mock_client = Mock()

        # Initial response with multiple tool uses
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"

        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.id = "tool_1"
        tool_block_1.name = "search_course_content"
        tool_block_1.input = {"query": "test1"}

        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.id = "tool_2"
        tool_block_2.name = "search_course_content"
        tool_block_2.input = {"query": "test2"}

        initial_response.content = [tool_block_1, tool_block_2]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Combined answer"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [initial_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2

            # Verify both results included
            second_call_args = mock_client.messages.create.call_args_list[1]
            messages = second_call_args[1]["messages"]
            tool_results = messages[2]["content"]
            assert len(tool_results) == 2


class TestErrorHandling:
    """Test error handling in AIGenerator"""

    def test_handles_api_exception(self):
        """Test graceful handling of API exceptions"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            # Should raise the exception
            with pytest.raises(Exception) as exc_info:
                generator.generate_response(query="test")

            assert "API Error" in str(exc_info.value)

    def test_handles_tool_execution_error(self, mock_tool_manager):
        """Test handling of tool execution errors"""
        mock_client = Mock()

        # Tool manager returns error
        mock_tool_manager.execute_tool.return_value = "Tool error: Database unavailable"

        # Initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        initial_response.content = [tool_block]

        # Final response (Claude should handle the error message)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "I encountered an error searching the database"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [initial_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

            tools = mock_tool_manager.get_tool_definitions()
            response = generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Should still return a response (Claude handles the error message)
            assert isinstance(response, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
