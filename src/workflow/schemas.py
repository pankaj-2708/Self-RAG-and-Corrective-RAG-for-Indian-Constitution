from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# NOTE: schema_for_retrieval_decider_node and schema_for_is_relevant_node have been
# removed — routing and relevance decisions are now handled by Jev (TypeSafe System-1
# model). The LLM is no longer called for those two nodes.


class schema_for_answer_from_context_node(BaseModel):
    response: str = Field(..., description="Response for given query")


parser_for_answer_from_context_node = PydanticOutputParser(
    pydantic_object=schema_for_answer_from_context_node
)


class schema_for_check_answer_grounded_node(BaseModel):
    """
    Used ONLY on the failure path of check_answer_grounded_node — i.e., when Jev has
    already determined the answer is NOT fully grounded. The LLM only needs to produce
    the actionable evidence text; the binary verdict is known (not_fully_supported).
    """

    evidence: str = Field(
        ...,
        description=(
            "Specific unsupported or incorrect claims from the answer, quoting them "
            "and explaining what is wrong or missing. Be specific and actionable so "
            "a revision agent can fix the issue."
        ),
    )


parser_for_schema_for_check_answer_grounded_node = PydanticOutputParser(
    pydantic_object=schema_for_check_answer_grounded_node
)


class schema_for_revise_answer_node(BaseModel):
    revised_response: str = Field(..., description="Response for given query")


parser_for_revise_answer_node = PydanticOutputParser(
    pydantic_object=schema_for_revise_answer_node
)


class schema_for_is_answer_relevant_node(BaseModel):
    """
    Used ONLY on the failure path of is_answer_relevant_node — i.e., when Jev has
    already determined the answer is NOT relevant. The LLM only needs to produce the
    actionable explanation text; the binary verdict is known (not relevant).
    """

    explanation: str = Field(
        ...,
        description=(
            "Detailed explanation of WHY the answer does not adequately address the query. "
            "Identify: which evaluation criteria it fails on, what aspects of the query are "
            "not addressed, and what a good answer should focus on. Be specific and actionable "
            "— this will be passed directly to a rewriting agent."
        ),
    )


parser_for_is_answer_relevant_node = PydanticOutputParser(
    pydantic_object=schema_for_is_answer_relevant_node
)


class schema_for_rewrite_answer_node(BaseModel):
    rewritten_response: str = Field(
        ...,
        description="Rewritten response that better addresses the user's query while remaining grounded in the provided contexts",
    )


parser_for_rewrite_answer_node = PydanticOutputParser(
    pydantic_object=schema_for_rewrite_answer_node
)


class schema_for_rewrite_answer_node(BaseModel):
    rewritten_response: str = Field(
        ...,
        description="Rewritten response that better addresses the user's query while remaining grounded in the provided contexts",
    )


parser_for_rewrite_answer_node = PydanticOutputParser(
    pydantic_object=schema_for_rewrite_answer_node
)


class RetrieverQueryItem(BaseModel):
    query: str = Field(..., description="Optimized search query string.")
    doc_type: Literal["IPC", "Constitution", "None"] = Field(
        ...,
        description="Whether a specific section/article is explicitly requested for this query (IPC or Constitution). Otherwise 'None'.",
    )
    number: Optional[str] = Field(
        None,
        description="The specific article or section number if explicitly requested (e.g. '21' or '302'). Otherwise null or empty.",
    )


class schema_for_retriever_query_node(BaseModel):
    retriever_queries: List[RetrieverQueryItem] = Field(
        ...,
        description="Optimized search queries for database retrieval. Generate only the required number of queries needed to answer the user query. Generate at most 3 queries.",
    )


parser_for_retriever_query_node = PydanticOutputParser(
    pydantic_object=schema_for_retriever_query_node
)


class schema_for_web_search_query_node(BaseModel):
    web_search_queries: List[str] = Field(
        ...,
        description="Optimized search queries for the web search engine. Generate at most 3 queries.",
    )


parser_for_web_search_query_node = PydanticOutputParser(
    pydantic_object=schema_for_web_search_query_node
)
