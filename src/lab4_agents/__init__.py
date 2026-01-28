"""
lab4_agents - A minimal AI agents framework

A didactic implementation showing how AI agents work with tools.
"""

from .agent import Agent
from .tool import Tool

# Make submodules accessible
from . import tools
from . import subagents

__all__ = [
    "Agent",
    "Tool",
    "tools",
    "subagents",
]
