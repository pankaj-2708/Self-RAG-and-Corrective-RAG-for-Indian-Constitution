import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

import os
import yaml
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from contextlib import asynccontextmanager

# ── Load workflow config ───────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)
_DEFAULT_DB_PATH = _cfg["workflow"]["db_path"]

from workflow.state import schema
from workflow.nodes import (
    retrieval_decider_node,
    retrieve_node,
    direct_generation_node,
    is_relevant_node,
    answer_from_context_node,
    check_answer_grounded_node,
    revise_answer_node,
    is_answer_relevant_node,
    rewrite_answer_node,
    web_search_node,
    generate_retriever_query_node,
    generate_web_search_query_node,
    fanout_relevant_node,
    aggregate_relevance,
    fanout_retrieve_node,
    aggregate_retrieval,
    memory_node,
    modify_short_term_memory_node,
)
from workflow.edges import (
    retrieval_decider_condition,
    is_relevant_condition,
    is_grounded_condition,
    is_answer_relevant_condition,
    memory_summary_condition,
)

if not os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGSMITH_TRACING_V2") == "false":
    try:
        from phoenix.otel import register

        tracer_provider = register(project_name="constitution", auto_instrument=True)
    except ImportError:
        pass

graph = StateGraph(state_schema=schema)

graph.add_node("retrieval_decider_node", retrieval_decider_node)
graph.add_node("generate_retriever_query_node", generate_retriever_query_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("direct_generation_node", direct_generation_node)
graph.add_node("is_relevant_node", is_relevant_node)
graph.add_node("answer_from_context_node", answer_from_context_node)
graph.add_node("check_answer_grounded_node", check_answer_grounded_node)
graph.add_node("revise_answer_node", revise_answer_node)
graph.add_node("is_answer_relevant_node", is_answer_relevant_node)
graph.add_node("rewrite_answer_node", rewrite_answer_node)
graph.add_node("generate_web_search_query_node", generate_web_search_query_node)
graph.add_node("web_search_node", web_search_node)
graph.add_node("aggregate_retrieval", aggregate_retrieval)
graph.add_node("aggregate_relevance", aggregate_relevance)
graph.add_node("memory_node", memory_node)
graph.add_node("modify_short_term_memory_node", modify_short_term_memory_node)

graph.add_edge(START, "retrieval_decider_node")
graph.add_conditional_edges(
    "retrieval_decider_node",
    retrieval_decider_condition,
    {
        "retrieval": "generate_retriever_query_node",
        "None": "direct_generation_node",
        "web_search": "generate_web_search_query_node",
    },
)
graph.add_conditional_edges(
    "generate_retriever_query_node",
    fanout_retrieve_node,
    path_map=["retrieve_node"],
)
graph.add_edge("generate_web_search_query_node", "web_search_node")
graph.add_edge("direct_generation_node", "memory_node")
graph.add_edge("retrieve_node", "aggregate_retrieval")
graph.add_conditional_edges(
    "aggregate_retrieval",
    fanout_relevant_node,
    path_map=["is_relevant_node"],
)
graph.add_edge("is_relevant_node", "aggregate_relevance")
graph.add_conditional_edges(
    "aggregate_relevance",
    is_relevant_condition,
    {True: "answer_from_context_node", False: "generate_web_search_query_node"},
)
graph.add_edge("web_search_node", "answer_from_context_node")
graph.add_edge("answer_from_context_node", "check_answer_grounded_node")
graph.add_conditional_edges(
    "check_answer_grounded_node",
    is_grounded_condition,
    {True: "is_answer_relevant_node", False: "revise_answer_node"},
)
graph.add_edge("revise_answer_node", "check_answer_grounded_node")
graph.add_conditional_edges(
    "is_answer_relevant_node",
    is_answer_relevant_condition,
    {True: "memory_node", False: "rewrite_answer_node"},
)
graph.add_edge("rewrite_answer_node", "is_answer_relevant_node")
graph.add_conditional_edges(
    "memory_node",
    memory_summary_condition,
    {"summarize": "modify_short_term_memory_node", "end": END},
)
graph.add_edge("modify_short_term_memory_node", END)


@asynccontextmanager
async def get_workflow(db_path: str = None):
    if db_path is None:
        db_path = _DEFAULT_DB_PATH
    async with AsyncSqliteSaver.from_conn_string(db_path) as ck_ptr:
        workflow = graph.compile(checkpointer=ck_ptr)
        yield workflow, ck_ptr


# Try compiling a checkpointer-less workflow just for drawing the mermaid graph
try:
    _temp_workflow = graph.compile()
    graph_png_bytes = _temp_workflow.get_graph().draw_mermaid_png()
    with open("workflow_image.png", "wb") as f:
        f.write(graph_png_bytes)
except Exception as e:
    print(f"Skipping workflow_image.png generation: {e}")
