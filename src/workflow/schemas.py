from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from langchain_core.output_parsers import PydanticOutputParser


class schema_for_retrieval_decider_node(BaseModel):
    retrieval_required: Literal["retrieval", "web_search", "None"] = Field(...)


parser_for_retrieval_decider_node = PydanticOutputParser(
    pydantic_object=schema_for_retrieval_decider_node
)


class schema_for_is_relevant_node(BaseModel):
    is_relevant_context: bool
    relevance_score: int = Field(
        ...,
        description=(
            "Relevance score from 0 to 10 indicating how strongly this context addresses the query. "
            "10 = perfectly addresses the exact question. "
            "0 = completely unrelated. "
            "Must always be provided regardless of is_relevant_context."
        ),
        ge=0,
        le=10,
    )


parser_for_is_relevant_node = PydanticOutputParser(
    pydantic_object=schema_for_is_relevant_node
)


class schema_for_answer_from_context_node(BaseModel):
    response: str = Field(..., description="Response for given query")


parser_for_answer_from_context_node = PydanticOutputParser(
    pydantic_object=schema_for_answer_from_context_node
)


class schema_for_check_answer_grounded_node(BaseModel):
    is_grounded: Literal["fully_supported", "not_fully_supported"]
    evidence: str = Field(
        ..., description="Proof that answer is not supported by given contexts"
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
    is_relevant: bool = Field(
        ...,
        description="Boolean indicating whether the answer is relevant to the user's query",
    )
    explanation: str = Field(
        default="",
        description="When is_relevant is false, a detailed explanation of why the answer is not relevant and what specific aspects need improvement. Empty when is_relevant is true.",
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
