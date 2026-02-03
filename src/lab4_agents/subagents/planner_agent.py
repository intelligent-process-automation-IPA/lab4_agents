import json
import re
from typing import Any

from ..agent import Agent
from ..tool import Tool


class PlannerAgent(Agent):
    """
    A Plan-and-Execute agent that breaks down complex tasks into steps.

    The agent works in three phases:
    1. Planning: Generates a multi-step plan
    2. Execution: Executes each step sequentially
    3. Re-planning: Evaluates completion and generates follow-up plans if needed
    """

    def __init__(
        self,
        model: str | None = None,
        tools: list[Tool] | None = None,
        executor_model: str | None = None,
        executor_tools: list[Tool] | None = None,
        max_replan_iterations: int = 3,
        system_prompt: str | None = None,
    ):
        """
        Initialize the PlannerAgent.

        Args:
            model: Model to use for planning (defaults to GROQ_MODEL env var)
            tools: List of tools available to the planner (optional, mainly for consistency)
            executor_model: Model to use for execution (defaults to same as planner)
            executor_tools: List of tools available to the executor agent
            max_replan_iterations: Maximum number of re-planning attempts (default: 3)
            system_prompt: Optional system message for the planner
        """
        # Initialize base Agent for planning
        planner_prompt = (
            system_prompt
            or "You are a planning assistant. Break down complex tasks into clear, "
            "sequential steps. Generate a numbered list of actionable steps. "
            "Each step should be specific and executable."
        )
        super().__init__(
            model=model,
            tools=tools or [],
            system_prompt=planner_prompt,
        )

        # Create executor agent
        executor_model = executor_model or self.model
        executor_tools = executor_tools or tools or []
        executor_prompt = (
            "You are an execution assistant. Execute the given step using available tools. "
            "Be precise and return clear results. If a step requires multiple tool calls, "
            "execute them in the correct order."
        )
        self.executor = Agent(
            model=executor_model,
            tools=executor_tools,
            system_prompt=executor_prompt,
        )

        self.max_replan_iterations = max_replan_iterations

        # Track plan and execution details for display
        self.last_plan: list[str] | None = None
        self.last_executed_steps: list[dict[str, Any]] | None = None
        self.last_replans: list[list[str]] | None = None

    def plan(self, user_query: str) -> list[str]:
        """
        Generate a multi-step plan for the given query.

        Args:
            user_query: The user's task or question

        Returns:
            A list of step strings representing the plan
        """
        planning_prompt = (
            f"Break down the following task into clear, sequential steps.\n\n"
            f"Task: {user_query}\n\n"
            f"Generate a numbered list of steps (1., 2., 3., etc.). "
            f"Each step should be a single, actionable instruction. "
            f"Be specific and clear about what needs to be done in each step."
        )

        # Use a temporary agent instance to avoid polluting main conversation
        temp_agent = Agent(
            model=self.model,
            tools=list(self.tools.values()),
            system_prompt=self.system_prompt,
        )
        response = temp_agent.run(planning_prompt)

        # Parse the response to extract plan steps
        steps = self._parse_plan(response)
        return steps

    def _parse_plan(self, plan_text: str) -> list[str]:
        """
        Parse plan text into a list of step strings.

        Args:
            plan_text: The LLM-generated plan text

        Returns:
            List of step strings
        """
        steps = []
        # Look for numbered list patterns: "1. Step", "2. Step", etc.
        # Also handle variations like "1)", "Step 1:", etc.
        patterns = [
            r"^\d+\.\s+(.+)$",  # "1. Step description"
            r"^\d+\)\s+(.+)$",  # "1) Step description"
            r"^Step\s+\d+[:\-]\s*(.+)$",  # "Step 1: description"
        ]

        lines = plan_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try each pattern
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    step = match.group(1).strip()
                    if step:
                        steps.append(step)
                    break
            else:
                # If no pattern matches but line looks like a step, include it
                # (handles cases where LLM doesn't use strict numbering)
                if len(steps) == 0 or len(line) > 10:  # Avoid very short lines
                    # Check if it's a continuation or a new step
                    if not line.startswith("-") and not line.startswith("*"):
                        steps.append(line)

        # If no steps were parsed, return the original text as a single step
        if not steps:
            steps = [plan_text.strip()]

        return steps

    def execute_step(
        self, step: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a single plan step using the executor agent.

        Args:
            step: The step description to execute
            context: Optional context from previous steps

        Returns:
            Dictionary with 'status', 'output', 'input', and optionally 'error'
        """
        # Build execution prompt with context if available
        if context:
            context_str = json.dumps(context, indent=2)
            execution_prompt = (
                f"Execute the following step:\n\n{step}\n\n"
                f"Context from previous steps:\n{context_str}\n\n"
                f"Use the available tools to complete this step."
            )
        else:
            execution_prompt = (
                f"Execute the following step:\n\n{step}\n\n"
                f"Use the available tools to complete this step."
            )

        try:
            # Reset executor to avoid state pollution between steps
            self.executor.reset()
            result = self.executor.run(execution_prompt)
            return {"status": "success", "output": result, "input": execution_prompt}
        except Exception as e:
            return {"status": "error", "output": None, "error": str(e), "input": execution_prompt}

    def replan(
        self,
        user_query: str,
        executed_steps: list[dict[str, Any]],
        current_plan: list[str],
    ) -> list[str] | None:
        """
        Evaluate if the task is complete and generate a new plan if needed.

        Args:
            user_query: Original user query
            executed_steps: List of executed step results
            current_plan: The plan that was just executed

        Returns:
            New plan if re-planning is needed, None if task is complete
        """
        # Build summary of executed steps
        step_summary = []
        for i, step_result in enumerate(executed_steps):
            step_num = i + 1
            status = step_result.get("status", "unknown")
            output = step_result.get("output", "")
            error = step_result.get("error", "")
            step_summary.append(
                f"Step {step_num}: {status}\n"
                f"Output: {output}\n"
                f"{f'Error: {error}' if error else ''}"
            )

        summary_text = "\n\n".join(step_summary)

        replanning_prompt = (
            f"Original task: {user_query}\n\n"
            f"Plan that was executed:\n"
            f"{chr(10).join(f'{i + 1}. {step}' for i, step in enumerate(current_plan))}\n\n"
            f"Execution results:\n{summary_text}\n\n"
            f"Evaluate whether the task has been completed successfully. "
            f"If the task is complete, respond with 'TASK_COMPLETE'. "
            f"If additional steps are needed, generate a new numbered list of steps to continue."
        )

        # Use a temporary agent for re-planning
        temp_agent = Agent(
            model=self.model,
            tools=list(self.tools.values()),
            system_prompt=self.system_prompt,
        )
        response = temp_agent.run(replanning_prompt)

        # Check if task is complete
        if (
            "TASK_COMPLETE" in response.upper()
            or "task is complete" in response.lower()
        ):
            return None

        # Parse new plan if provided
        new_steps = self._parse_plan(response)
        if new_steps:
            return new_steps

        # If no clear plan but also no completion, assume we need to continue
        # Return None to indicate we should finish (safety fallback)
        return None

    def run(self, user_query: str, max_iterations: int = 10) -> str:
        """
        Main entry point: Plan, execute, and re-plan as needed.

        Args:
            user_query: The user's task or question
            max_iterations: Unused, kept for API compatibility with Agent.run

        Returns:
            Final response to the user
        """
        # Generate initial plan
        plan = self.plan(user_query)
        if not plan:
            return "Error: Could not generate a plan for the task."

        # Store initial plan
        self.last_plan = plan
        self.last_replans = []

        replan_count = 0
        all_executed_steps: list[dict[str, Any]] = []

        while replan_count <= self.max_replan_iterations:
            # Execute all steps in the current plan
            executed_steps: list[dict[str, Any]] = []
            context: dict[str, Any] = {}

            for step in plan:
                step_result = self.execute_step(step, context)
                executed_steps.append(step_result)

                # Update context with step result
                if step_result.get("status") == "success":
                    context[f"step_{len(executed_steps)}"] = step_result.get(
                        "output", ""
                    )

            # Store executed steps
            self.last_executed_steps = executed_steps

            # Add executed steps to overall history
            all_executed_steps.extend(executed_steps)

            # Check if we need to re-plan
            if replan_count < self.max_replan_iterations:
                new_plan = self.replan(user_query, executed_steps, plan)
                if new_plan is None:
                    # Task is complete, generate final response
                    break
                # Update plan and continue
                plan = new_plan
                self.last_replans.append(plan)
                replan_count += 1
            else:
                # Max replan iterations reached
                break

        # Generate final response summarizing the results
        final_prompt = (
            f"Original task: {user_query}\n\n"
            f"All executed steps and their results:\n"
            f"{json.dumps(all_executed_steps, indent=2)}\n\n"
            f"Provide a clear, concise final answer to the user's task."
        )

        # Use executor to generate final response (it has access to tools if needed)
        self.executor.reset()
        final_response = self.executor.run(final_prompt)
        return final_response
