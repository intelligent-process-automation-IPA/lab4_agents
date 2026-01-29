"""
Specialized calculator agent.

A pre-configured agent with calculator tools.
"""

from ..agent import Agent
from ..tools.calculator import calculator_tool


class CalculatorAgent(Agent):
    """A specialized agent pre-configured with calculator tools."""

    def __init__(self, model: str | None = None):
        """
        Initialize the calculator agent.

        Args:
            model: Model to use (defaults to GROQ_MODEL env var)
        """
        super().__init__(
            model=model,
            tools=calculator_tool,
            system_prompt="You are a helpful calculator assistant. "
            "Use the available tools to perform calculations.",
        )
