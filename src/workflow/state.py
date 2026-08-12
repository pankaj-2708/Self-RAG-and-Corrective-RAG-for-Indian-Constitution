from typing import TypedDict, List, Literal, Optional, Annotated
import operator
from pydantic import Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

def deduplicate_reducer(existing: List[str], new: List[str]) -> List[str]:
    if existing is None:
        existing = []
    if new is None:
        new = []

    # to reset after turn
    if len(new)!=0 and new[0] == "-1":
        return []

    return list(dict.fromkeys(existing + new))


class schema(TypedDict):
    retrieval_required: Literal["retrieval", "web_search", "None"]
    web_searched: bool
    user_query: str
    retriever_queries: Optional[List[dict]]
    web_search_queries: Optional[List[str]]
    retrieved_contexts: Annotated[List[str], deduplicate_reducer]
    relevant_contexts: Annotated[List[str], deduplicate_reducer]
    answer_for_query: str
    generated_response: str
    is_grounded: Literal["fully_supported", "not_fully_supported"]
    is_supported: bool
    is_answer_relevant: bool
    relevance_explanation: str
    evidence: str
    k: Optional[int] = Field(default=3)
    max_retry_for_groundness_checking: Optional[int] = Field(default=3)
    max_retry_for_answer_relevant_checking: Optional[int] = Field(default=2)
    messages: List[BaseMessage]
    local_memory_summary: Optional[str]
    max_turns_before_summarisation: Optional[int]
    turns_until_summary_update: Optional[int]
    messages_to_include: Optional[int]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
