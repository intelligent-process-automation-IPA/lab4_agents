"""
Simple example of using the agent framework.

This demonstrates a basic single-shot query to the agent.
For interactive TUI mode, use main-tui.py instead.
"""

from lab4_agents.agent import Agent
from lab4_agents.tools import calculator


def main():
    """Run a simple agent example."""
    # Create an agent with calculator tools
    agent = Agent(
        tools=[calculator.calculator_tool],
    )

    # Ask a simple question
    print("🤖 Agent Example")
    print("-" * 40)
    response = agent.run("What is 10 + 20?")
    print(f"Response: {response}")
    print()

    # Ask a follow-up (same agent instance keeps conversation history)
    response = agent.run("What about 100 * 5?")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
