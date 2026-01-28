"""
Calculator tools for AI agents.

Provides basic arithmetic operations as tools that agents can use.
"""

from pydantic import BaseModel, Field
from typing import Literal, Any

from ..tool import Tool


class CalculatorParams(BaseModel):
    """Defines the expected arguments for a calculator tool."""

    x: float = Field(description="First number to use in the operation")
    y: float = Field(description="Second number to use in the operation")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The operation to perform. Options: add, subtract, multiply, divide"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"operation": "add", "x": 2, "y": 3},
                {"operation": "subtract", "x": 5, "y": 2},
                {"operation": "multiply", "x": 3, "y": 4},
                {"operation": "divide", "x": 10, "y": 2},
            ]
        }
    }


def calculator(args: dict[str, Any]) -> dict[str, Any]:
    """Perform basic arithmetic operations."""
    try:
        params = CalculatorParams(**args)
        if params.operation == "add":
            return {"result": params.x + params.y}
        elif params.operation == "subtract":
            return {"result": params.x - params.y}
        elif params.operation == "multiply":
            return {"result": params.x * params.y}
        elif params.operation == "divide":
            return {"result": params.x / params.y}
        else:
            return {"error": "Invalid operation"}
    except Exception as e:
        return {"error": f"Error performing operation: {str(e)}"}


# Create tools
calculator_tool = Tool(
    name="calculator",
    description="Perform basic arithmetic operations",
    parameters=CalculatorParams,
    func=calculator,
)

__all__ = ["calculator_tool"]
