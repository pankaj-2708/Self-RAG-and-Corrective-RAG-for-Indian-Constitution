# src

Core Python source for the Self-RAG pipeline — the LangGraph workflow, CLI, vector store builder, and evaluation suite.

## Structure

```
src/
├── cli.py                   # Rich-powered interactive terminal interface
├── create_vector_store.py   # Ingests JSON corpus → ChromaDB vector store
├── workflow/                # LangGraph state machine (all nodes, edges, state)
└── evaluation/              # Ragas evaluation pipeline
```

## Entry Points

### CLI

```bash
python src/cli.py
# optional: resume a specific thread
python src/cli.py --thread_id <uuid>
```

An interactive terminal chat powered by [Rich](https://github.com/Textualize/rich). Streams node-by-node progress, renders Markdown responses, and shows token usage + latency after each turn.

| Command | Action |
|---|---|
| `/new` | Start a fresh conversation thread |
| `exit` | Quit the CLI |

### Build the Vector Store

Run once to embed and persist the legal corpus into ChromaDB:

```bash
python src/create_vector_store.py
```

Reads `data/articles.json` (Constitution articles) and `data/penal_code_sections.json` (IPC sections), embeds them with `sentence-transformers/all-mpnet-base-v2`, and writes to `data/constitution_and_ipc.chroma/`.

## Sub-packages

| Package | README |
|---|---|
| [`workflow/`](workflow/README.md) | LangGraph graph definition — nodes, edges, state, prompts, schemas |
| [`evaluation/`](evaluation/README.md) | End-to-end Ragas evaluation — knowledge graph, clustering, test set, scoring |
