"""
PlannerAgent example showcasing plan-and-execute capability.

This demonstrates the PlannerAgent which breaks down complex tasks into steps,
executes them sequentially, and re-plans if needed.
"""

from lab4_agents.subagents.planner_agent import PlannerAgent
from lab4_agents.tools import calculator


def main():
    """Run a PlannerAgent example."""
    # Create a planner agent with calculator tools
    planner = PlannerAgent(
        executor_tools=[calculator.calculator_tool],
    )

    # Example 1: Simple multi-step calculation
    print("🤖 PlannerAgent Example 1: Multi-step Calculation")
    print("-" * 50)
    query = "Calculate the following: First add 15 and 25, then multiply the result by 2"
    print(f"Query: {query}\n")
    response = planner.run(query)
    print(f"Response: {response}")
    print()

    # Example 2: Complex task requiring multiple steps
    print("🤖 PlannerAgent Example 2: Complex Task")
    print("-" * 50)
    query = (
        "I need to figure out the total cost of items. "
        "Item A costs $30, Item B costs $45. "
        "Calculate the sum, then multiply by 1.1 to account for tax. "
        "Finally, divide by 2 to get the per-person cost for a split."
    )
    print(f"Query: {query}\n")
    response = planner.run(query)
    print(f"Response: {response}")
    print()

    # Example 3: Demonstrating re-planning capability
    print("🤖 PlannerAgent Example 3: Multi-step with Validation")
    print("-" * 50)
    query = (
        "Create a plan to: "
        "1) Calculate 100 divided by 5, "
        "2) Add 50 to the result, "
        "3) Multiply by 3, "
        "4) Check if the final result is greater than 400"
    )
    print(f"Query: {query}\n")
    response = planner.run(query)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
