"""
Specialized calculator agent.

A pre-configured agent with calculator tools.
"""

from ..agent import Agent
from ..tools.calculator import CALCULATOR_TOOLS


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
            tools=CALCULATOR_TOOLS,
            system_prompt="You are a helpful calculator assistant. "
            "Use the available tools to perform calculations.",
        )
