"""
An interactive terminal UI that maintains conversation history and allows
the user to send multiple messages to the agent in a loop.
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from lab4_agents.agent import Agent
from lab4_agents.tools import calculator


class TUI:
    """
    Interactive TUI for continuous agent operation.

    Maintains agent conversation history and allows multi-turn interaction
    with slash-command support for reset, help, and exit.
    """

    def __init__(self, agent: Agent | None = None, verbose: bool = False):
        """
        Initialize the TUI and agent.

        Args:
            agent: Optional pre-configured Agent instance. If None, a default
                   agent with calculator tools will be created.
            verbose: If True, display tool calls and their arguments/results
        """
        self.console = Console()
        self.verbose = verbose
        if agent is not None:
            self.agent = agent
        else:
            # Default: Create an agent with calculator tools
            self.agent = Agent(
                tools=[calculator.calculator_tool],
            )

    def _print_header(self) -> None:
        """Print welcome banner."""
        title = Text("🤖 Agent TUI", style="bold cyan")
        self.console.print(Panel(title, expand=False, border_style="cyan"))
        self.console.print(
            Text(
                "Type your question or command. Type '/help' for available commands.",
                style="dim italic",
            )
        )
        self.console.print()

    def _print_help(self) -> None:
        """Print available commands."""
        self.console.print(Rule("Available Commands", style="cyan"))
        help_text = """
[cyan]/help[/cyan]    Show this help message
[cyan]/reset[/cyan]   Clear conversation history (keeps system prompt)
[cyan]/verbose[/cyan] Toggle verbose mode (shows tool calls)
[cyan]/quit[/cyan]    Exit the TUI
[cyan]/exit[/cyan]    Same as /quit
        """.strip()
        self.console.print(help_text)
        self.console.print()

    def _handle_command(self, line: str) -> bool:
        """
        Handle slash commands.

        Args:
            line: The user input line

        Returns:
            True if should quit, False otherwise
        """
        line_lower = line.strip().lower()

        if line_lower in ("/quit", "/exit"):
            self.console.print(Text("Goodbye!", style="cyan"))
            return True

        if line_lower == "/reset":
            self.agent.reset()
            self.console.print(Text("✓ Conversation history cleared.", style="green"))
            return False

        if line_lower == "/help":
            self._print_help()
            return False

        if line_lower == "/verbose":
            self.verbose = not self.verbose
            status = "enabled" if self.verbose else "disabled"
            self.console.print(Text(f"✓ Verbose mode {status}.", style="green"))
            return False

        # Unknown command
        self.console.print(
            Text(f"Unknown command: {line}", style="yellow"),
        )
        self.console.print(Text("Type '/help' for available commands.", style="dim"))
        return False

    def _extract_tool_calls(self) -> list[dict]:
        """
        Extract tool calls from the agent's message history.

        Returns:
            List of tool call information dictionaries
        """
        tool_calls_info = []

        for message in self.agent.messages:
            if message.get("role") == "assistant" and "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tool_calls_info.append(
                        {
                            "tool": tool_name,
                            "arguments": arguments,
                        }
                    )
            # If a tool message follows, attach the result
            elif message.get("role") == "tool" and tool_calls_info:
                result = json.loads(message.get("content", "{}"))
                tool_calls_info[-1]["result"] = result

        return tool_calls_info

    def _format_tool_calls(self, tool_calls: list[dict]) -> str:
        """
        Format tool calls as a readable string for display in a panel.

        Args:
            tool_calls: List of tool call information

        Returns:
            Formatted string representation of tool calls
        """
        if not tool_calls:
            return ""

        lines = []
        for i, call in enumerate(tool_calls, 1):
            tool_name = call["tool"]
            arguments = call["arguments"]
            result = call.get("result", {})

            lines.append(f"[yellow]Tool {i}:[/yellow] [cyan]{tool_name}[/cyan]")
            lines.append(f"  [dim]Arguments:[/dim]")
            for key, value in arguments.items():
                lines.append(f"    {key}: {value}")

            if result:
                lines.append(f"  [dim]Result:[/dim]")
                for key, value in result.items():
                    lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def _send(self, user_input: str) -> None:
        """
        Send user input to agent and display response.

        Args:
            user_input: The user's message
        """
        try:
            # Show user input
            user_panel = Panel(
                user_input,
                title="You",
                border_style="blue",
                expand=False,
            )
            self.console.print(user_panel)

            # Record message count before to detect new tool calls
            messages_before = len(self.agent.messages)

            # Get agent response
            response = self.agent.run(user_input)

            # Extract and show tool calls in verbose mode (before agent response)
            if self.verbose:
                new_tool_calls = []
                for message in self.agent.messages[messages_before:]:
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            arguments = json.loads(tool_call["function"]["arguments"])
                            new_tool_calls.append(
                                {
                                    "tool": tool_name,
                                    "arguments": arguments,
                                }
                            )
                    # Attach result to the last tool call
                    elif message.get("role") == "tool" and new_tool_calls:
                        result = json.loads(message.get("content", "{}"))
                        new_tool_calls[-1]["result"] = result

                # Display tool calls in a panel if any were made
                if new_tool_calls:
                    tool_calls_text = self._format_tool_calls(new_tool_calls)
                    tool_panel = Panel(
                        tool_calls_text,
                        title="Tool Calls",
                        border_style="yellow",
                        expand=False,
                    )
                    self.console.print(tool_panel)

            # Show agent response
            agent_panel = Panel(
                response,
                title="Agent",
                border_style="green",
                expand=False,
            )
            self.console.print(agent_panel)

        except Exception as e:
            # Show error
            error_text = f"Error: {str(e)}"
            error_panel = Panel(
                error_text,
                title="Error",
                border_style="red",
                style="red",
            )
            self.console.print(error_panel)

        self.console.print()  # Spacing

    def run(self) -> None:
        """Main TUI loop."""
        self._print_header()

        while True:
            try:
                # Get user input with styled prompt
                user_input = input("You: ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Check for commands
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                    continue

                # Send regular message to agent
                self._send(user_input)

            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl-C and Ctrl-D gracefully
                self.console.print(Text("\nGoodbye!", style="cyan"))
                break
