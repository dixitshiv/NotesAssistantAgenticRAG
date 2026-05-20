"""Tools the agent can call.

Three tools, each for a different question shape:
  - search_notes(q)        — semantic search across chunk content
  - find_notes_by_name(q)  — substring match on filenames (brand/name lookup)
  - list_note_files()      — enumerate all note filenames
"""

from pathlib import Path

from langchain_core.tools import tool

from . import config, indexing


# Build the vector store once at import time and keep a module-level
# retriever. Cheap — no embedding happens here, just opening the on-disk
# collection.
_vstore = indexing.build_vstore()
_retriever = _vstore.as_retriever(search_kwargs={"k": 8})


@tool
def search_notes(query: str) -> str:
    """Semantic search over the content of the user's notes. Use this for
    conceptual questions ("how does X work", "what's the trade-off between
    A and B", "explain the architecture"). NOT good for finding a note by
    its name/title — use find_notes_by_name for that."""
    hits = _retriever.invoke(query)
    if not hits:
        return "No matching notes."
    return "\n\n".join(
        f"[{Path(d.metadata['source']).name}]\n{d.page_content}"
        for d in hits
    )


@tool
def find_notes_by_name(name: str) -> str:
    """Find notes whose FILENAME contains the given substring (case-insensitive).
    Use this when the user asks about a specific project/brand/title that
    is likely a filename, e.g. "What is TransferIQ?" or "Tell me about
    BioInsight". Returns the full content of every matching note."""
    notes_dir = Path(config.require_notes_dir()).expanduser().resolve()
    needle = name.lower().replace(" ", "")   # tolerate spaces in the query

    matches: list[Path] = []
    for path in sorted(notes_dir.rglob("*.md")):
        stem = path.stem.lower().replace(" ", "").replace("_", "")
        if needle in stem:
            matches.append(path)

    if not matches:
        return f"No note filenames contain {name!r}."

    return "\n\n".join(
        f"[{p.name}]\n{p.read_text(encoding='utf-8', errors='replace')}"
        for p in matches
    )


@tool
def list_note_files() -> str:
    """List the filenames of every note. Use this when the user asks
    "what notes do I have", "how many readmes", or any question about
    enumeration/coverage of the notes collection."""
    notes_dir = Path(config.require_notes_dir()).expanduser().resolve()
    files = sorted(notes_dir.rglob("*.md"))
    if not files:
        return "No notes found."
    return f"{len(files)} note(s):\n" + "\n".join(f"- {p.name}" for p in files)
