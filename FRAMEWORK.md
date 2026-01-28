# Minimal AI Agents Framework

A didactic, minimal implementation of an AI agent framework using the OpenAI API.

## Overview

This framework demonstrates how to build agents that can use tools. The agent runs in a loop:

1. **Accept user input** - Get a question or task
2. **Call OpenAI API** - Send messages and available tools to the model
3. **Check response** - Does the model want to use a tool?
4. **Execute tools** - If yes, run the tool and capture results
5. **Feed back results** - Send tool results back to the model
6. **Repeat** - Continue until the model provides a final answer

## Core Components

### `Tool`
Represents a tool the agent can use. Contains:
- `name` - Tool identifier
- `description` - What the tool does (sent to OpenAI)
- `parameters` - JSON schema describing input parameters
- `func` - Python function to execute

### `Agent`
Main agent class that:
- Manages conversation history with the model
- Calls OpenAI API
- Executes tools when requested
- Handles tool results

## Key Design Principles

**Simplicity First**: This implementation prioritizes clarity over features. It's meant to be educational.

**Direct API calls**: No abstraction layers - you see exactly how the OpenAI API works.

**Tool loop**: The agent automatically handles the request/response cycle for tool use.

## Usage Example

```python
from lab4_agents import Agent, Tool

# Define a tool
def add(input_dict: dict) -> dict:
    return {"result": input_dict["a"] + input_dict["b"]}

add_tool = Tool(
    name="add",
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    },
    func=add,
)

# Create agent with tools
agent = Agent(
    model="gpt-4o-mini",
    tools=[add_tool],
    system_prompt="You are a calculator assistant.",
)

# Run the agent
response = agent.run("What is 5 + 3?")
print(response)
```

## File Structure

- `agent.py` - Core Agent and Tool classes
- `example.py` - Calculator agent example
- `FRAMEWORK.md` - This documentation

## How It Works

### Message Flow

1. User sends: `"What is 5 + 3?"`
2. Agent adds to messages: `{"role": "user", "content": "What is 5 + 3?"}`
3. Agent calls OpenAI with available tools
4. OpenAI responds: `{"tool_calls": [{"name": "add", "arguments": {"a": 5, "b": 3}}]}`
5. Agent adds to messages: OpenAI's response with `tool_calls`
6. Agent executes the tool: `add({"a": 5, "b": 3})` → `{"result": 8}`
7. Agent adds to messages: `{"role": "tool", "content": "{"result": 8}"}`
8. Agent calls OpenAI again with all messages
9. OpenAI responds: `"5 + 3 = 8"`
10. Agent returns this to user

### The Loop

The `Agent.run()` method implements this loop:

```python
while True:
    response = call_openai_api()

    if no tool_calls in response:
        return response_text

    for each tool_call:
        result = execute_tool(tool_call)
        add_result_to_messages(result)
```

## Why This Matters

This framework shows:
- How modern AI agents work
- The tool/function calling capability in LLMs
- How to integrate tools with language models
- The request-response cycle for agent loops

It's intentionally simple so you can understand every line and extend it as needed.
