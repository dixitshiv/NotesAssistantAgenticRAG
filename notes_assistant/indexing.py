"""Read markdown notes from disk and keep Chroma in sync.

Two modes:
  - load_notes()        : just read files, no embedding.
  - index_incremental() : diff disk vs Chroma, only re-embed what changed.
"""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    """Cheap content fingerprint. SHA-256 over the raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_into_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_notes() -> list[Document]:
    """Read every *.md under NOTES_DIR. Used by load-only diagnostics."""
    notes_dir = Path(config.require_notes_dir()).expanduser().resolve()
    if not notes_dir.is_dir():
        raise SystemExit(f"NOTES_DIR is not a directory: {notes_dir}")

    docs: list[Document] = []
    for path in sorted(notes_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": str(path)},
        ))
    return docs


# ---------------------------------------------------------------------------
# Vector store handle
# ---------------------------------------------------------------------------

def build_vstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    return Chroma(
        collection_name="notes",
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------

def _scan_disk() -> dict[str, str]:
    """Walk NOTES_DIR. Return {file_id (abs path): file_hash}."""
    notes_dir = Path(config.require_notes_dir()).expanduser().resolve()
    return {
        str(path): _file_hash(path)
        for path in sorted(notes_dir.rglob("*.md"))
        if path.read_text(encoding="utf-8", errors="replace").strip()
    }


def _scan_chroma(vstore: Chroma) -> dict[str, str]:
    """Read all chunks' metadata and collapse to {file_id: file_hash}.

    Many chunks per file all share the same (file_id, file_hash), so this
    deduplicates naturally.
    """
    data = vstore.get(include=["metadatas"])
    state: dict[str, str] = {}
    for meta in data["metadatas"] or []:
        fid = meta.get("file_id")
        fhash = meta.get("file_hash")
        if fid and fhash:
            state[fid] = fhash
    return state


def _embed_file(vstore: Chroma, file_id: str, file_hash: str) -> int:
    """Read a file, chunk it, embed it, store with deterministic chunk ids."""
    path = Path(file_id)
    text = path.read_text(encoding="utf-8", errors="replace")
    pieces = _split_into_chunks(text)

    filename = path.name
    docs = []
    ids = []
    for i, piece in enumerate(pieces):
        chunk_id = f"{file_id}::chunk_{i}"
        docs.append(Document(
            page_content=f"[FILE: {filename}]\n{piece}",
            metadata={
                "source": file_id,
                "file_id": file_id,
                "file_hash": file_hash,
                "chunk_index": i,
            },
        ))
        ids.append(chunk_id)

    if docs:
        vstore.add_documents(docs, ids=ids)
    return len(docs)


def index_incremental(vstore: Chroma) -> dict[str, int]:
    """Sync the vector store to match disk. Only re-embeds what changed.

    Returns a summary dict {added, updated, deleted, unchanged}.
    """
    disk = _scan_disk()
    existing = _scan_chroma(vstore)

    to_delete = set(existing) - set(disk)
    to_add = set(disk) - set(existing)
    to_update = {fid for fid in set(disk) & set(existing) if disk[fid] != existing[fid]}
    unchanged = set(disk) & set(existing) - to_update

    # Delete first (also clears stale chunks for updates).
    for fid in to_delete | to_update:
        vstore.delete(where={"file_id": fid})

    # Then add new or updated files.
    for fid in to_add | to_update:
        _embed_file(vstore, fid, disk[fid])

    return {
        "added": len(to_add),
        "updated": len(to_update),
        "deleted": len(to_delete),
        "unchanged": len(unchanged),
    }


if __name__ == "__main__":
    vstore = build_vstore()
    summary = index_incremental(vstore)
    print(
        f"Indexed: {summary['added']} added, "
        f"{summary['updated']} updated, "
        f"{summary['deleted']} deleted, "
        f"{summary['unchanged']} unchanged."
    )
