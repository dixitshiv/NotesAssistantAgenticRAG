"""Middleware: cross-cutting hooks that run inside the agent loop.

Right now just one — a logger that prints what's happening before each
model call and after each model response. Useful for understanding what
the agent did per turn (which tools it picked, how it phrased the query,
how big the message list is getting).
"""

from typing import Any

from langchain.agents.middleware import AgentState, after_model, before_model
from langgraph.runtime import Runtime


@before_model
def log_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]
    print(f"  [before_model] n={len(state['messages'])} last={type(last).__name__}")
    return None


@after_model
def log_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        for tc in tool_calls:
            print(f"  [after_model]  tool: {tc['name']}({tc['args']})")
    else:
        preview = (last.text or "").strip()[:80].replace("\n", " ")
        print(f"  [after_model]  text: {preview!r}")
    return None
