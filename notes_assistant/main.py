"""Entry point: index notes (if needed), then run the REPL with memory."""

# Load .env BEFORE importing anything that reads env vars (config, langsmith, etc).
from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from . import config, indexing  # noqa: E402
from .agent import build_agent  # noqa: E402


def sync_index() -> None:
    """Diff disk vs Chroma. Only re-embed changed files."""
    vstore = indexing.build_vstore()
    s = indexing.index_incremental(vstore)
    print(
        f"Indexed: {s['added']} added, {s['updated']} updated, "
        f"{s['deleted']} deleted, {s['unchanged']} unchanged."
    )


def repl(agent, thread_id: str) -> None:
    invoke_config = {"configurable": {"thread_id": thread_id}}
    print(f"Ready (thread='{thread_id}'). Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q in {"", "exit", "quit"}:
            return
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            invoke_config,
        )
        print("bot>", result["messages"][-1].text, "\n")


def main() -> None:
    config.require_notes_dir()
    sync_index()
    with SqliteSaver.from_conn_string(config.DB_PATH) as checkpointer:
        agent = build_agent(checkpointer=checkpointer)
        repl(agent, config.THREAD_ID)


if __name__ == "__main__":
    main()
