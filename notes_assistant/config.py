"""Single source of truth for paths, model names, and constants.

Kept in one file so the rest of the project can import settings without
reaching for os.environ scattered everywhere.
"""

import os


# --- Models --------------------------------------------------------------
MODEL = "ollama:gemma4:31b-cloud"
EMBED_MODEL = "nomic-embed-text"

# --- Where things live on disk ------------------------------------------
NOTES_DIR = os.environ.get("NOTES_DIR")   # required; checked at startup
CHROMA_DIR = "./chroma_db"                # persistent vector store
DB_PATH = "./assistant_memory.db"         # SQLite checkpointer (v3)

# --- Memory --------------------------------------------------------------
THREAD_ID = os.environ.get("THREAD_ID", "main")   # override via env for fresh sessions


def require_notes_dir() -> str:
    """Resolve NOTES_DIR or fail loudly. Called at the top of every entry point."""
    if not NOTES_DIR:
        raise SystemExit(
            "NOTES_DIR is not set. Export it first, e.g.\n"
            "  export NOTES_DIR=~/Documents/notes"
        )
    return NOTES_DIR
