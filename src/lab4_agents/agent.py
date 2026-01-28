"""
Minimal AI Agents Framework

A didactic implementation showing how AI agents work with tools.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from .tool import Tool

# Load environment variables from .env file
load_dotenv()

# Load GROQ configuration from environment variables
GROQ_API_ENDPOINT = os.getenv("GROQ_API_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")


class Agent:
    """
    A simple AI agent that can use tools.

    The agent works in a loop:
    1. Send messages to OpenAI
    2. If the model wants to use tools, execute them
    3. Send tool results back to the model
    4. Repeat until we get a final text response
    """

    def __init__(
        self,
        model: str | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the agent.

        Args:
            model: Model to use (defaults to GROQ_MODEL env var or "llama-3.1-70b-versatile")
            tools: List of tools the agent can use
            system_prompt: Optional system message to set agent behavior
        """
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_ENDPOINT)
        self.model = model or GROQ_MODEL
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = []

        # Add system prompt if provided
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _get_openai_tools(self) -> list[dict[str, Any]] | None:
        """Get tools in OpenAI format."""
        if not self.tools:
            return None
        return [tool.to_openai_format() for tool in self.tools.values()]

    def _call_api(self) -> ChatCompletionMessage:
        """Make a single API call to OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self._get_openai_tools(),
        )
        return response.choices[0].message

    def _execute_tool_call(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> dict[str, Any]:
        """Execute a single tool call and return the result."""
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # Find and execute the tool
        tool = self.tools.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            result = tool.execute(arguments)
            return result
        except Exception as e:
            return {"error": str(e)}

    def _process_tool_calls(self, message: ChatCompletionMessage) -> None:
        """Process all tool calls in a message."""
        # First, add the assistant's message with tool calls
        # Only include necessary fields, exclude annotations which API doesn't support
        assistant_message: dict[str, Any] = {
            "role": "assistant",
        }
        if message.content:
            assistant_message["content"] = message.content
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        self.messages.append(assistant_message)

        # Then execute each tool and add results
        for tool_call in message.tool_calls:
            result = self._execute_tool_call(tool_call)

            # Add tool result to messages
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    def run(self, user_input: str) -> str:
        """
        Run the agent with user input.

        This is the main agent loop:
        1. Add user message
        2. Call OpenAI API
        3. If tool calls, execute them and loop back
        4. If text response, return it

        Args:
            user_input: The user's message/question

        Returns:
            The agent's final text response
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Agent loop
        while True:
            # Call the API
            message = self._call_api()

            # Check if the model wants to use tools
            if message.tool_calls:
                # Execute tools and add results to messages
                self._process_tool_calls(message)
                # Continue the loop to get the next response
                continue

            # No tool calls - we have a final response
            # Add assistant message to history
            self.messages.append({"role": "assistant", "content": message.content})
            return message.content

    def reset(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
