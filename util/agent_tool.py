"""Global tool registry and @agent_tool decorator.

Import this module anywhere tools are defined.  The decorator auto-registers
functions into TOOL_REGISTRY so agent_runtime can discover them at graph-build time.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    name: str
    description: str
    func: Callable
    param_model: type[BaseModel] | None = None


TOOL_REGISTRY: dict[str, ToolEntry] = {}


def agent_tool(
    name: str,
    description: str,
    param_model: type[BaseModel] | None = None,
) -> Callable:
    """Register a function as an agent-callable tool.

    Args:
        name: Tool name exposed to the LLM.
        description: Tool description (passed to the LLM).
        param_model: Optional Pydantic model for the tool's input schema.
    """
    def decorator(func: Callable) -> Callable:
        TOOL_REGISTRY[name] = ToolEntry(
            name=name,
            description=description,
            func=func,
            param_model=param_model,
        )
        logger.info("Tool registered: %s", name)
        return func

    return decorator


def get_registered_tools() -> list[ToolEntry]:
    return list(TOOL_REGISTRY.values())
