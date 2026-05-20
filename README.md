# Notes Assistant

A CLI agent that lets you chat with your own markdown notes. Persistent vector
search, three sharply-scoped tools, cross-session memory, and full observability
through LangSmith.

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LangChain 1.0 (`create_agent`) |
| Orchestration runtime | LangGraph |
| LLM | Ollama (`gemma4:31b-cloud`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector store | Chroma (persistent, local disk) |
| Conversation memory | LangGraph SQLite checkpointer |
| Observability | LangSmith (optional) |
| Config / secrets | `python-dotenv` |
| Package manager | `uv` |

---

## Architecture

```
                          ┌──────────────────────────────┐
   .md files on disk ───► │  index_incremental           │
                          │  (sha256 diff, embed deltas) │
                          └──────────┬───────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────────┐
                          │  Chroma (./chroma_db)        │
                          │  per-chunk metadata:         │
                          │  file_id, file_hash, source  │
                          └──────────┬───────────────────┘
                                     │
   ┌─────────┐    user msg           ▼
   │  REPL   │ ──────────► ┌──────────────────────────────┐
   └────▲────┘             │  create_agent                │
        │                  │   ├── tools: search_notes,   │
        │ AIMessage        │   │           find_notes_    │
        │                  │   │           by_name,       │
        │                  │   │           list_note_     │
        │                  │   │           files          │
        │                  │   ├── middleware: log_*      │
        │                  │   └── checkpointer: SQLite   │
        │                  └──────────┬───────────────────┘
        │                             │
        │              all turns ◄────┤ thread_id="main"
        │              persisted      │
        │                             ▼
        │                  ┌──────────────────────────────┐
        │                  │  ./assistant_memory.db       │
        │                  │  (SQLite — full message      │
        │                  │   history per thread)        │
        │                  └──────────────────────────────┘
        │
        └── (optional) every call traced to LangSmith
```

---

## File layout

```
notes_assistant/
├── config.py         constants: model name, paths, thread id, env guard
├── indexing.py       load notes → chunk → embed → Chroma; incremental diff
├── tools.py          the three tools the agent can call
├── middleware.py     @before_model / @after_model logging hooks
├── agent.py          create_agent assembly + system prompt
├── main.py           entry point: dotenv → sync index → REPL with memory
└── README.md         (this file)
```

Each file is ~25–120 lines. One concept per file.

---

## Tools the agent has

| Tool | Purpose | When the agent picks it |
|---|---|---|
| `search_notes(query)` | Semantic vector search over chunk content | "How does X work", "trade-off between A and B", explanatory |
| `find_notes_by_name(name)` | Substring match on filenames | "What is BrandX?", "Tell me about ProjectY" |
| `list_note_files()` | Enumerate every note's filename | "How many notes", "what files do I have" |

Tool docstrings cross-reference each other (e.g. `search_notes`'s docstring says
"use `find_notes_by_name` for name-based lookups"). That phrasing is the routing
logic — the model picks based on the docstrings, no separate router.

---

## Key Technical Decisions

**Why three tools instead of one super-tool?**
A single tool ("search everything semantically") is the textbook RAG pattern,
but vector search is genuinely bad at brand-name lookups and enumeration.
Adding a filename-substring tool and a list tool fixed two real failure modes
that no amount of prompt-tuning or `k` adjustment would solve. The lesson:
*add capability, don't bolt on prompt rules*.

**Why prepend `[FILE: filename]` to every chunk before embedding?**
Vector search embeds chunk *content*, not filenames. A query like "TransferIQ"
wouldn't surface chunks from `README_TransferIQ.md` because the brand name
appears mostly in the filename, not the body. Injecting the filename into the
embedded text gives the brand name weight in similarity scoring.

**Why Chroma, not FAISS or pgvector?**
Personal-scale corpus, local-first, persistent to disk with one line of config,
and supports metadata filters (`vstore.delete(where={"file_id": ...})`) which
the incremental indexer relies on. FAISS would need manual persistence.
pgvector would need a server. Chroma is the right zero-friction default.

**Why incremental indexing instead of "if non-empty, skip"?**
The "skip if non-empty" shortcut means edits and deletions never propagate.
With incremental sync — hash every file, diff against stored metadata, only
re-embed deltas — the index always reflects what's actually on disk. Costs
~milliseconds on the no-change case (the common case).

**Why SQLite for conversation memory, not in-memory?**
Conversations survive process restart. `thread_id="main"` means re-running
the CLI tomorrow continues yesterday's conversation, including the model's
recall of past questions and prior tool results.

**Why `python-dotenv` instead of just exporting env vars?**
The `.env` file keeps LangSmith API key and `NOTES_DIR` in one place. The
import order matters — `load_dotenv()` runs *before* any `os.environ.get(...)`
in `config.py`, so secrets reach the process. Anything imported above
`load_dotenv()` in `main.py` would see an empty environment.

**Why an `agent.py` that's only ~35 lines?**
Composition over flexibility. The file's only job is to wire model + tools +
middleware + checkpointer + system prompt into `create_agent(...)`. Everything
substantive lives elsewhere. If `agent.py` ever grows past ~50 lines,
something's misplaced.

---

## Run it

```bash
# One-time setup — installs deps from pyproject.toml + uv.lock
uv sync
ollama pull nomic-embed-text
ollama pull gemma4:31b-cloud   # see "Swapping the model" below

# Configure
cat > .env <<EOF
NOTES_DIR=/path/to/your/markdown/folder
# Optional — for LangSmith tracing:
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=PersonalRAG
EOF

# Run
uv run python -m notes_assistant.main
```

Override the conversation thread (for testing or per-session isolation):

```bash
THREAD_ID=test-foo uv run python -m notes_assistant.main
```

### Swapping the model

The chat model is hardcoded in `config.py`:

```python
MODEL = "ollama:gemma4:31b-cloud"
```

Change that one line to point at any model `init_chat_model` recognises —
for example `"ollama:llama3.2"` (local) or `"anthropic:claude-sonnet-4-6"`
(set `ANTHROPIC_API_KEY` in `.env` first). The embedding model
(`nomic-embed-text`) is independent and lives in the same file.

The model needs to support **tool calling** — most modern chat models do,
but tiny local models often don't. If `find_notes_by_name` never gets
called, the model probably isn't tool-capable.

---

## What you'll see at startup

```
Indexed: 0 added, 0 updated, 0 deleted, 8 unchanged.
Ready (thread='main'). Type 'exit' or Ctrl-D to quit.

you> What is TransferIQ?
  [before_model] n=1 last=HumanMessage
  [after_model]  tool: find_notes_by_name({'name': 'TransferIQ'})
  [before_model] n=3 last=ToolMessage
  [after_model]  text: 'TransferIQ is an agentic platform ...'
bot> TransferIQ is an agentic platform designed to automate technology
     transfer ... (from README_TransferIQ.md).
```

The `[before_model]` / `[after_model]` lines are middleware emitting the agent
loop state. Each user turn produces one or more model calls; each model call
either decides on a tool or produces a final answer.

---

## On-disk artifacts

| Path | Purpose | Safe to delete? |
|---|---|---|
| `chroma_db/` | Vector store (chunk embeddings + metadata) | Yes — next run re-embeds everything |
| `assistant_memory.db` | SQLite conversation checkpoints | Yes — wipes all conversation memory |
| `.env` | Secrets + `NOTES_DIR` | No — git-ignored anyway |

---

## Known limits (intentional)

- **Conversation memory and corpus can diverge.** If you delete a note that the
  agent already cited in a previous turn, the agent may still recall the content
  from message history. Use a fresh `THREAD_ID` after destructive corpus changes
  if you need clean tests.
- **No streaming.** `agent.stream()` is supported but flaky on Ollama cloud
  models. Local Ollama models stream fine.
- **No re-ranking.** Top-`k=8` chunks go straight to the model. A real
  production system might add a cross-encoder reranker between retrieval and
  generation.
- **Read-only.** The agent never writes/edits notes.
- **Single user, single thread.** No multi-tenant routing.

---

## Where this fits in the LangChain 1.0 mental model

Every piece of this project maps to one primitive from the framework:

| Concept | Where it lives here |
|---|---|
| Chat model + messages | `config.MODEL` passed through `create_agent` |
| `@tool` + tool calling | `tools.py` |
| Agentic RAG (retriever-as-tool) | `search_notes` |
| Structured output | (not used — final answer is conversational) |
| Memory via checkpointer + `thread_id` | `main.py` (SqliteSaver context) |
| Middleware | `middleware.py` |
| Streaming | (deferred — Ollama-cloud limitation) |
| Observability | LangSmith env vars in `.env` |

If you want to extend it: add a tool, add a middleware, swap the model in
`config.py`. The architecture absorbs change without rewrites.
