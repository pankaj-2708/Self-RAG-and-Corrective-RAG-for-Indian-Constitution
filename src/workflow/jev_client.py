"""
jev_client.py
Thin async wrapper around TypeSafe Jev for all decision nodes in the Self-RAG workflow.

Jev is a System-1 model: non-autoregressive, ~70–500 ms per call, returns typed structured
decisions (Choice, Noul, Score) rather than free-form text. It replaces full LLM calls for
pure classification/routing tasks in the workflow.

Reference: https://python.langchain.com/docs/integrations/providers/typesafe/
"""

import os
import asyncio
from functools import lru_cache

from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier


@lru_cache(maxsize=1)
def _get_classifier() -> TypeSafeClassifier:
    """Lazily initialise (and cache) a single TypeSafeClassifier instance."""
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise ValueError(
            "TYPESAFE_API_KEY is not set. Add it to your .env file."
        )
    return TypeSafeClassifier(api_key=api_key)


async def jev_route_query(user_query: str) -> str:
    """
    Classify the user query into one of three retrieval routes.

    Uses Jev Choice — selects exactly one option from a predefined criteria set.

    Returns:
        "retrieval"   — fetch from local vector DB (Constitution / IPC text)
        "web_search"  — fetch from the web (current events, case law, etc.)
        "None"        — answer directly, no retrieval needed (greetings, chitchat)
    """
    clf = _get_classifier()
    response = await asyncio.to_thread(
        clf.invoke,
        {
            "state": f"User query: {user_query}",
            "questions": {
                "route": Choice(
                    instructions=(
                        "You are routing queries for a legal Q&A system about the Indian "
                        "Constitution and IPC (Indian Penal Code). "
                        "Classify the query into exactly one of the three routes below."
                    ),
                    criteria={
                        "retrieval": (
                            "The query asks about the definition, text, wording, punishment, "
                            "scope, rights, or duties in a specific IPC Section or Constitutional "
                            "Article — including comparisons or reasoning about provisions already "
                            "in the static legal corpus. Also use this when the query asks you to "
                            "compare, contrast, or reason about the relationship between provisions "
                            "that are fully contained in the Constitution or IPC."
                        ),
                        "web_search": (
                            "The query requires current events, recent Supreme Court / High Court "
                            "judgments, ongoing proceedings, proposed but not yet enacted "
                            "amendments, or any information beyond the static enacted text."
                        ),
                        "None": (
                            "The query is a greeting, casual remark, or can be answered with "
                            "general knowledge — no legal document retrieval or web search needed."
                        ),
                    },
                )
            },
        },
    )
    return response.choices["route"].choice  # "retrieval" | "web_search" | "None"


async def jev_check_relevance(
    user_query: str,
    context: str,
    threshold: float = 0.6,
) -> tuple[bool, int]:
    """
    Decide whether a retrieved context chunk is relevant to the query.

    Uses Jev Noul (yes/no probability) + Score (0–100 rating).
    Score is normalized to 0–10 to match the existing schema.

    Returns:
        (is_relevant: bool, relevance_score: int 0–10)
    """
    clf = _get_classifier()
    response = await asyncio.to_thread(
        clf.invoke,
        {
            "state": (
                f"User Query:\n{user_query}\n\n"
                f"Retrieved Legal Context:\n{context}"
            ),
            "questions": {
                "is_relevant": Noul(
                    instructions=(
                        "Does this retrieved legal context directly address or substantially "
                        "help answer the user's query? "
                        "Return high probability if the chunk covers the specific section, "
                        "article, right, punishment, or legally adjacent provision the query "
                        "asks about. Return low probability if it is from an entirely different "
                        "area of law or only tangentially connected."
                    )
                ),
                "relevance_score": Score(
                    instructions=(
                        "Rate the relevance of this context to the user's query. "
                        "High = directly answers the exact section/article asked. "
                        "Low = completely unrelated or tangential."
                    )
                ),
            },
        },
    )
    noul_prob: float = response.nouls["is_relevant"].noul
    raw_score: float = response.scores["relevance_score"].score  # 0–100
    # Normalize to 0–10 to match the existing schema_for_is_relevant_node
    score_0_10: int = round(raw_score / 10)
    return noul_prob >= threshold, score_0_10


async def jev_check_grounding(
    answer: str,
    context: str,
    threshold: float = 0.6,
) -> bool:
    """
    Decide whether the generated answer is fully supported by the retrieved contexts.

    Uses Jev Noul (yes/no probability).

    Returns:
        True  → "fully_supported"      (caller skips LLM — happy path)
        False → "not_fully_supported"  (caller invokes LLM only to get evidence text)
    """
    clf = _get_classifier()
    response = await asyncio.to_thread(
        clf.invoke,
        {
            "state": (
                f"Generated Answer:\n{answer}\n\n"
                f"Retrieved Legal Contexts:\n{context}"
            ),
            "questions": {
                "is_grounded": Noul(
                    instructions=(
                        "Is every factual claim, legal citation, section/article reference, "
                        "punishment detail, and right/duty stated in the answer directly "
                        "and explicitly supported by the provided contexts? "
                        "Return high probability only if ALL claims have clear textual evidence "
                        "in the contexts. Any unsupported fact, number, or citation should "
                        "lower the probability significantly."
                    )
                ),
            },
        },
    )
    return response.nouls["is_grounded"].noul >= threshold


async def jev_check_answer_relevance(
    user_query: str,
    answer: str,
    threshold: float = 0.6,
) -> bool:
    """
    Decide whether the generated answer actually addresses the user's query.

    Uses Jev Noul (yes/no probability).

    Returns:
        True  → answer is relevant      (caller skips LLM — happy path)
        False → answer is not relevant  (caller invokes LLM only to get explanation text)
    """
    clf = _get_classifier()
    response = await asyncio.to_thread(
        clf.invoke,
        {
            "state": (
                f"User Query:\n{user_query}\n\n"
                f"Generated Answer:\n{answer}"
            ),
            "questions": {
                "is_answer_relevant": Noul(
                    instructions=(
                        "Does the generated answer directly and completely address what the "
                        "user asked? "
                        "Return high probability if the answer covers all parts of the query, "
                        "addresses the correct legal provision, and provides meaningful legal "
                        "information rather than a generic or evasive response. "
                        "Return low probability if the answer addresses the wrong provision, "
                        "ignores major parts of the query, or only states that information "
                        "is unavailable without any useful content."
                    )
                ),
            },
        },
    )
    return response.nouls["is_answer_relevant"].noul >= threshold
