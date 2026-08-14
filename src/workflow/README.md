# workflow

The LangGraph state machine that implements the full Self-RAG / Corrective-RAG pipeline. This package is the heart of the project — every retrieval decision, relevance check, grounding verification, and self-correction loop lives here.

## Files

| File | Description |
|---|---|
| [`__init__.py`](__init__.py) | Assembles the `StateGraph`, wires all nodes and edges, compiles the graph with an `AsyncSqliteSaver` checkpointer, and exports `get_workflow()` — an async context manager that yields `(workflow, checkpointer)`. Also saves `workflow_image.png` at import time. |
| [`state.py`](state.py) | `TypedDict` state schema (`schema`) with all fields and their reducer functions (e.g., `add_messages` for conversation history, `operator.add` for retrieved context lists). |
| [`nodes.py`](nodes.py) | All 13+ node implementations — each is an async function that reads from and writes to the graph state. |
| [`edges.py`](edges.py) | Conditional routing functions used by `add_conditional_edges` — decide which node to visit next based on state values. |
| [`config.py`](config.py) | Initialises shared resources: DeepSeek R1/V3 LLM clients (via AWS Bedrock), the ChromaDB vector store, the Tavily search tool, and the HuggingFace embedding model. |
| [`prompts.py`](prompts.py) | System prompt strings for every node. Keeping prompts separate from node logic makes iteration easy. |
| [`schemas.py`](schemas.py) | Pydantic models used for structured LLM output (e.g., routing decision, relevance verdict, grounding verdict). |

## Graph Topology

```
START
  └─► retrieval_decider_node
        ├─[retrieval]──► generate_retriever_query_node
        │                  └─[fan-out]──► retrieve_node (parallel)
        │                                  └──► aggregate_retrieval
        │                                         └─[fan-out]──► is_relevant_node (parallel)
        │                                                          └──► aggregate_relevance
        │                                                                ├─[relevant]──► answer_from_context_node
        │                                                                └─[not relevant]──► generate_web_search_query_node
        ├─[web_search]─► generate_web_search_query_node
        │                  └──► web_search_node
        │                         └──► answer_from_context_node
        └─[direct]─────► direct_generation_node
                                └──► memory_node

answer_from_context_node
  └──► check_answer_grounded_node
         ├─[grounded]────► is_answer_relevant_node
         │                    ├─[relevant]──► memory_node
         │                    │                ├─[summarize]──► modify_short_term_memory_node ──► END
         │                    │                └─[end]──────────────────────────────────────────► END
         │                    └─[not relevant]──► rewrite_answer_node ──► is_answer_relevant_node
         └─[not grounded]──► revise_answer_node ──► check_answer_grounded_node
```

## Usage

```python
from workflow import get_workflow

async with get_workflow() as (workflow, ck_ptr):
    async for chunk in workflow.astream(
        {"user_query": "What is Article 21?", "k": 2, ...},
        {"configurable": {"thread_id": "some-uuid"}},
        stream_mode="updates",
    ):
        print(chunk)
```
