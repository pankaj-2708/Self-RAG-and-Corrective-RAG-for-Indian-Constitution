
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from workflow.state import schema
from workflow.nodes import (
    retrieval_decider_node, retrieve_node, direct_generation_node,
    is_relevant_node, answer_from_context_node, check_answer_grounded_node,
    revise_answer_node, is_answer_useful_node, rewrite_query_node, web_search_node,
    generate_retriever_query_node, generate_web_search_query_node,fanout_relevant_node,aggregate_relevance
)
from workflow.edges import (
    retrieval_decider_condition, is_relevant_condition,
    is_grounded_condition, is_answer_useful_condition
)

from phoenix.otel import register
tracer_provider = register(
  project_name="constitution",
  auto_instrument=True 
)

conn = sqlite3.connect("statedb.db", check_same_thread=False)
ck_ptr = SqliteSaver(conn=conn)

graph = StateGraph(state_schema=schema)

graph.add_node("retrieval_decider_node", retrieval_decider_node)
graph.add_node("generate_retriever_query_node", generate_retriever_query_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("direct_generation_node", direct_generation_node)
graph.add_node("is_relevant_node", is_relevant_node)
graph.add_node("answer_from_context_node", answer_from_context_node)
graph.add_node("check_answer_grounded_node", check_answer_grounded_node)
graph.add_node("revise_answer_node", revise_answer_node)
graph.add_node("is_answer_useful_node", is_answer_useful_node)
graph.add_node("rewrite_query_node", rewrite_query_node)
graph.add_node("generate_web_search_query_node", generate_web_search_query_node)
graph.add_node("web_search_node", web_search_node)
graph.add_node("aggregate_relevance", aggregate_relevance)

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
graph.add_edge("generate_retriever_query_node", "retrieve_node")
graph.add_edge("generate_web_search_query_node", "web_search_node")
graph.add_edge("direct_generation_node", END)
graph.add_conditional_edges("retrieve_node", fanout_relevant_node)
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
    {True: "is_answer_useful_node", False: "revise_answer_node"},
)
graph.add_edge("revise_answer_node", "is_answer_useful_node")
graph.add_conditional_edges(
    "is_answer_useful_node",
    is_answer_useful_condition,
    {True: END, False: "rewrite_query_node"},
)
graph.add_edge("rewrite_query_node", "retrieval_decider_node")

workflow = graph.compile(checkpointer=ck_ptr)

# graph_png_bytes = workflow.get_graph().draw_mermaid_png()

# with open("workflow_image.png", "wb") as f:
#     f.write(graph_png_bytes)
