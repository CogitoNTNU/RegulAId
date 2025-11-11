"""Streaming callback handler for LangChain agents."""

import asyncio
import importlib
from typing import Any, Dict, List


# Dynamically import LangChain classes to support multiple versions
def _get_async_callback_handler():
    """Dynamically locate AsyncCallbackHandler across LangChain versions."""
    candidates = [
        'langchain_core.callbacks.base',
        'langchain.callbacks.base',
        'langchain_core.callbacks',
        'langchain.callbacks',
    ]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            handler = getattr(mod, 'AsyncCallbackHandler', None)
            if handler is not None:
                return handler
        except Exception:
            continue
    raise ImportError('Could not locate AsyncCallbackHandler in langchain')


def _get_agent_types():
    """Dynamically locate AgentAction and AgentFinish types."""
    candidates = [
        'langchain_core.agents',
        'langchain.schema.agent',
        'langchain.schema',
        'langchain_core.messages',
    ]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            action = getattr(mod, 'AgentAction', None)
            finish = getattr(mod, 'AgentFinish', None)
            if action is not None and finish is not None:
                return action, finish
        except Exception:
            continue
    # Fallback: return Any type hints if not found
    return Any, Any


AsyncCallbackHandler = _get_async_callback_handler()
AgentAction, AgentFinish = _get_agent_types()


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming agent execution steps."""

    def __init__(self):
        """Initialize the streaming callback handler."""
        self.queue = asyncio.Queue()
        self.finished = False

    async def on_agent_start(self, serialized: Dict[str, Any], input: str, **kwargs) -> None:
        """Called when agent starts."""
        await self.queue.put({
            "type": "task",
            "status": "in_progress",
            "title": "Starting analysis",
            "description": "Initializing AI agent"
        })

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")

        # Map tool names to user-friendly descriptions
        tool_descriptions = {
            "retrieve_eu_ai_act": {
                "title": "Searching EU AI Act database",
                "description": f"Looking for relevant articles and regulations"
            },
            "retrieve_risk_requirements": {
                "title": "Analyzing risk requirements",
                "description": f"Fetching compliance requirements for specified risk level"
            },
            "retrieve_system_type_info": {
                "title": "Searching system-specific information",
                "description": f"Finding requirements for {input_str[:50]}"
            }
        }

        task_info = tool_descriptions.get(tool_name, {
            "title": f"Running {tool_name}",
            "description": f"Processing: {input_str[:100]}"
        })

        await self.queue.put({
            "type": "task",
            "status": "in_progress",
            "title": task_info["title"],
            "description": task_info["description"],
            "tool": tool_name,
            "input": input_str
        })

    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing."""
        # Extract article count from output if available
        article_count = output.count("[") if "[" in output else 0

        await self.queue.put({
            "type": "task",
            "status": "completed",
            "title": "Search completed",
            "description": f"Retrieved {article_count} relevant articles" if article_count > 0 else "Search completed"
        })

    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error."""
        await self.queue.put({
            "type": "task",
            "status": "error",
            "title": "Tool execution failed",
            "description": str(error)
        })

    async def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent decides on an action."""
        # This is called after tool selection but before execution
        # We already handle this in on_tool_start, so we can skip
        pass

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating."""
        # Don't send a task update here, the agent is just thinking
        pass

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token."""
        # NOTE: Token streaming is disabled for agent executors because:
        # 1. The agent uses tools and reasoning loops, so raw tokens aren't meaningful
        # 2. Streaming tokens can cause API errors in complex agent contexts
        # 3. We stream task progress instead, which is more useful for this use case
        pass

    async def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM finishes generating."""
        # LLM finished, don't need to signal anything special
        pass

    async def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes."""
        await self.queue.put({
            "type": "task",
            "status": "completed",
            "title": "Analysis complete",
            "description": "Finalizing classification"
        })

        # Signal that we're done
        self.finished = True

    async def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when chain encounters an error."""
        await self.queue.put({
            "type": "error",
            "message": str(error)
        })
        self.finished = True

    async def get_updates(self):
        """Generator that yields updates as they become available."""
        while not self.finished or not self.queue.empty():
            try:
                # Wait for updates with a timeout
                update = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                yield update
            except asyncio.TimeoutError:
                # No update available, continue waiting
                if self.finished and self.queue.empty():
                    break
                continue
