"""Assemble the agent.

This file's only job is to compose the pieces:
  model + tools + system prompt  -->  create_agent
Each piece lives elsewhere (config.py, tools.py, etc.). Keeping the
assembly thin means every concept stays in its own file.
"""

from langchain.agents import create_agent

from . import config
from .middleware import log_after, log_before
from .tools import find_notes_by_name, list_note_files, search_notes


SYSTEM_PROMPT = """\
You are the user's personal notes assistant. You ONLY answer based on the
content of their notes.

Rules:
- For any question, call `search_notes` first with a focused query.
- Answer ONLY from what the tool returns.
- If the notes don't cover the question, reply exactly:
  "That's not in your notes."
- Never answer from your own general knowledge.
- Cite the source filename for any fact you state, like (from foo.md).
"""


def build_agent(checkpointer=None):
    """Construct the agent. Returns a compiled LangGraph agent.

    If a checkpointer is passed, conversation state persists across
    invocations (and across process restarts if the checkpointer itself
    is backed by disk). Without one, every invoke() is a fresh turn.
    """
    return create_agent(
        model=config.MODEL,
        tools=[search_notes, find_notes_by_name, list_note_files],
        system_prompt=SYSTEM_PROMPT,
        middleware=[log_before, log_after],
        checkpointer=checkpointer,
    )
