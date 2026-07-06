from typing import TypedDict, List, Literal, Optional,Annotated
from pydantic import Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class schema(TypedDict):
    retrieval_required: Literal["retrieval", "web_search", "None"]
    web_searched: bool
    user_query: str
    retriever_queries: Optional[List[dict]]
    web_search_queries: Optional[List[str]]
    retrieved_contexts: List[str]
    relevant_contexts: Annotated[List[str],add_messages]
    answer_for_query: str
    generated_response: str
    is_grounded: bool
    is_supported: bool
    is_answer_useful: bool
    evidence: str
    k: Optional[int] = Field(default=3)
    max_retry_for_revise_answer: Optional[int] = Field(default=3)
    max_retry_for_rewrite_query: Optional[int] = Field(default=2)
    messages: List[BaseMessage]
