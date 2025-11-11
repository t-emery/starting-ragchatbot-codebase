from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Course Outline Tool**: Use for queries about course structure, lessons, or what's covered in a course
  - Returns: Course title, course link, instructor, and complete list of lessons with numbers and titles
  - Use when user asks: "What's in this course?", "Show me the outline", "What lessons are available?", etc.
  - **IMPORTANT**: When using this tool, you MUST include the course link in your response

- **Course Content Search Tool**: Use for questions about specific course content or detailed educational materials
  - Returns: Relevant content chunks from lessons with context
  - Use when user asks about specific topics, concepts, or lesson details

- **Multi-Round Tool Usage**:
  - You can make **UP TO 2 SEQUENTIAL tool calls** per user query
  - **First round**: Gather preliminary information (e.g., get course outline to learn lesson structure)
  - **Second round**: Use that information to make more targeted searches (e.g., search specific lesson content)
  - **Example workflow**: "Get outline of Course X → Learn lesson 4 is about Y → Search courses about topic Y"
  - After each tool call, you'll receive results and can decide whether another tool call would help
  - If you have enough information after one tool call, provide your answer directly

- **When to use multiple rounds**:
  - User asks comparative questions requiring multiple lookups
  - Need course structure first, then specific content
  - Need to identify a lesson topic, then search for related courses
  - **Do NOT** use multiple rounds if one search gives sufficient information

- **Tool Synthesis**:
  - Choose the most appropriate tool for each round
  - Synthesize all tool results into accurate, fact-based responses
  - If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course structure questions**: Use the outline tool and include the course link URL
- **Course content questions**: Use the search tool
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the search results" or "I used the outline tool"
  - Exception: Always include course links when provided by the outline tool

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_tool_rounds: int = None,
    ) -> str:
        """
        Generate AI response with optional multi-round tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds (default: from config)

        Returns:
            Generated response as string
        """
        from config import config  # Import here to avoid circular dependency

        # Use config value if not explicitly provided
        if max_tool_rounds is None:
            max_tool_rounds = config.MAX_TOOL_ROUNDS

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get initial response from Claude
        response = self.client.messages.create(**api_params)

        # Handle multi-round tool execution if needed
        if tools and tool_manager:
            response = self._handle_multi_round_tool_execution(
                response=response,
                base_params=api_params,
                tool_manager=tool_manager,
                max_rounds=max_tool_rounds,
            )

        # Extract and return final text response
        return self._extract_text_response(response)

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(content_block.name, **content_block.input)

                tool_results.append(
                    {"type": "tool_result", "tool_use_id": content_block.id, "content": tool_result}
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {**self.base_params, "messages": messages, "system": base_params["system"]}

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _extract_text_response(self, response) -> str:
        """
        Extract text content from response, handling potential multi-block responses.

        Args:
            response: Anthropic API response object

        Returns:
            Extracted text content
        """
        for content_block in response.content:
            if hasattr(content_block, "text"):
                return content_block.text

        # Fallback: if no text block found, return empty string
        return ""

    def _force_final_response(
        self, response, messages: List[Dict[str, Any]], base_params: Dict[str, Any], tool_manager
    ):
        """
        Force a final text response when max rounds reached but tool_use still active.

        This executes the pending tools but makes the final API call WITHOUT tools
        to ensure we get a text response.

        Args:
            response: Response with tool_use blocks
            messages: Accumulated message history (mutated in place)
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final API response with text content
        """
        # Add assistant's tool use to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute pending tool calls
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(content_block.name, **content_block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": content_block.id, "content": tool_result}
                )

        # Add results to messages
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Make final call WITHOUT tools to force text response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
            # NOTE: Deliberately omitting "tools" parameter
        }

        final_response = self.client.messages.create(**final_params)
        return final_response

    def _execute_tool_round(
        self,
        response,
        messages: List[Dict[str, Any]],
        base_params: Dict[str, Any],
        tool_manager,
        round_number: int,
        is_final_round: bool,
    ):
        """
        Execute a single round of tool calling and get next response.

        Args:
            response: Current response with tool_use blocks
            messages: Accumulated message history (mutated in place)
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            round_number: Current round number (1-indexed for logging)
            is_final_round: Whether this is the last allowed round

        Returns:
            Next API response (may or may not have tool_use)
        """
        # Add assistant's tool use response to message history
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls in this response
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                # Execute the tool
                tool_result = tool_manager.execute_tool(content_block.name, **content_block.input)

                # Add to results
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": content_block.id, "content": tool_result}
                )

        # Add tool results to message history
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare next API call parameters
        next_params = {**self.base_params, "messages": messages, "system": base_params["system"]}

        # KEY DECISION: Include tools UNLESS this is the final round
        # This allows Claude to make another tool call if needed
        if not is_final_round and "tools" in base_params:
            next_params["tools"] = base_params["tools"]
            next_params["tool_choice"] = {"type": "auto"}

        # Make next API call
        next_response = self.client.messages.create(**next_params)

        return next_response

    def _handle_multi_round_tool_execution(
        self, response, base_params: Dict[str, Any], tool_manager, max_rounds: int
    ):
        """
        Handle multi-round tool execution with continuation pattern.

        This method allows Claude to make up to max_rounds sequential tool calls,
        where each round can use information from previous rounds.

        Args:
            response: Initial API response (may contain tool_use)
            base_params: Base API parameters including system prompt and initial message
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds allowed

        Returns:
            Final API response object (with stop_reason != "tool_use")

        Flow:
            Round 0: Initial response (already made)
            Round 1-N: Execute tools → Build messages → API call with tools
            Final: When stop_reason != "tool_use" or max_rounds reached
        """
        # Start building message history from initial user message
        messages = base_params["messages"].copy()
        current_response = response
        round_count = 0

        # Continuation loop: keep processing while we have tool calls and haven't exceeded rounds
        while current_response.stop_reason == "tool_use" and round_count < max_rounds:

            # Execute one round of tool calling
            current_response = self._execute_tool_round(
                response=current_response,
                messages=messages,
                base_params=base_params,
                tool_manager=tool_manager,
                round_number=round_count + 1,
                is_final_round=(round_count == max_rounds - 1),
            )

            round_count += 1

        # Handle case where max rounds reached but Claude still wants to use tools
        if current_response.stop_reason == "tool_use" and round_count >= max_rounds:
            # Make final call WITHOUT tools to force a text response
            current_response = self._force_final_response(
                response=current_response,
                messages=messages,
                base_params=base_params,
                tool_manager=tool_manager,
            )

        return current_response
