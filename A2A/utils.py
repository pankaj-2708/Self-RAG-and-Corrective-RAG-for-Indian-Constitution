def _extract_node_details(node_name: str, data: dict) -> dict:
    """Pick out the useful, user-facing fields from a node's output."""
    if not isinstance(data, dict):
        return {}

    match node_name:
        case "retrieval_decider_node":
            return {"route": data.get("retrieval_required")}

        case "generate_retriever_query_node":
            queries = data.get("retriever_queries") or []
            return {
                "retriever_queries": [
                    {
                        "query": q.get("query"),
                        "doc_type": q.get("doc_type"),
                        "number": q.get("number"),
                    }
                    for q in queries
                    if isinstance(q, dict)
                ]
            }

        case "retrieve_node":
            contexts = data.get("retrieved_contexts") or []
            return {
                "retrieved_count": len(contexts),
                "retrieved_previews": [_preview_text(c, 120) for c in contexts],
                "retrieved_titles": [_context_title(c) for c in contexts],
            }

        case "is_relevant_node":
            retrieved = data.get("retrieved_contexts") or []
            rel = data.get("relevant_contexts") or []
            kept_indices = []
            for passage in rel:
                if not isinstance(passage, str):
                    continue
                for index, candidate in enumerate(retrieved):
                    if candidate == passage:
                        kept_indices.append(index)
                        break
            return {
                "marked_relevant": len(rel) > 0,
                "kept_indices": kept_indices,
            }

        case "aggregate_retrieval":
            return {}

        case "generate_web_search_query_node":
            return {"web_queries": data.get("web_search_queries") or []}

        case "web_search_node":
            ctxs = data.get("relevant_contexts") or []
            titles = []
            for c in ctxs:
                if not isinstance(c, str):
                    continue
                for line in c.split("\n"):
                    line = line.strip()
                    if line.lower().startswith("title -"):
                        titles.append(line[len("title -") :].strip())
                        break
            return {
                "web_result_count": len(ctxs),
                "web_titles": titles,
            }

        case "answer_from_context_node":
            return {"generated": True}

        case "direct_generation_node":
            return {"generated": True}

        case "check_answer_grounded_node":
            return {
                "is_grounded": data.get("is_grounded"),
                "evidence_preview": _preview_text(data.get("evidence") or "", 150),
            }

        case "revise_answer_node":
            return {"revised": True}

        case "is_answer_relevant_node":
            return {
                "is_relevant": data.get("is_answer_relevant"),
                "explanation_preview": _preview_text(
                    data.get("relevance_explanation") or "",
                    150,
                ),
            }

        case "rewrite_answer_node":
            return {"rewritten": True}

        case _:
            return {}