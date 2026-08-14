from fastapi import FastAPI
import sys
import os
import json
import asyncio
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import yaml
from langchain_core.messages import messages_to_dict

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)
from src.workflow import get_workflow

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/all_chats")
async def get_all_chats():
    """
    Returns all chat threads with their chat_id and formatted messages.
    Only the latest checkpoint per thread_id is included.
    Threads are ordered by the most recent checkpoint (newest first), so a
    thread that was just appended to floats to the top of the list.
    """
    try:
        results = []

        async with get_workflow() as (workflow, ck_ptr):
            # Pick the latest checkpoint row per thread_id, then order those
            # by the same checkpoint_id descending. LangGraph's checkpoint_id
            # starts with a sortable timestamp prefix, so this gives us
            # most-recently-updated first.
            async with ck_ptr.conn.execute(
                """
                SELECT thread_id, MAX(checkpoint_id) AS latest_checkpoint
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY latest_checkpoint DESC
                """
            ) as cursor:
                thread_ids = [row[0] async for row in cursor]

            for thread_id in thread_ids:
                state = await workflow.aget_state(
                    config={"configurable": {"thread_id": thread_id}}
                )
                raw_messages = state.values.get("messages", [])
                formatted_messages = [
                    {"role": msg["type"], "content": msg["data"].get("content", "")}
                    for msg in messages_to_dict(raw_messages)
                ]
                results.append({"chat_id": thread_id, "messages": formatted_messages})

        return {"status": "success", "response": results}
    except Exception as e:
        print(e)
        return {"status": "failed", "error": str(e)}

# Load config parameters
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)
_pipeline_defaults = _cfg.get("pipeline_defaults", {})


def _preview_text(value, limit=120):
    if not isinstance(value, str):
        return value
    return value[:limit] + ("…" if len(value) > limit else "")


def _context_title(value):
    """Pull a short heading out of a retrieved passage.

    The corpus tends to lead with a line like ``Title - Article 14`` or
    ``title - Article 14``. If we don't find one we fall back to the first
    non-empty line so the ledger can still label the passage.
    """
    if not isinstance(value, str):
        return ""
    for line in value.split("\n"):
        cleaned = line.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered.startswith("title -") or lowered.startswith("article -"):
            return cleaned.split(" - ", 1)[-1].strip() or cleaned
        return cleaned[:80]
    return ""


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
                        titles.append(line[len("title -"):].strip())
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


async def run_workflow(thread_id, user_query):
    try:
        async with get_workflow() as (workflow, ck_ptr):
            existing = await ck_ptr.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            if existing:
                # if conv already exists
                initial_state = {
                    "user_query": user_query,
                    "k": _pipeline_defaults.get("k", 2),
                    "max_retry_for_groundness_checking": _pipeline_defaults.get("max_retry_for_groundness_checking", 1),
                    "max_retry_for_answer_relevant_checking": _pipeline_defaults.get("max_retry_for_answer_relevant_checking", 1),
                }
            else:
                initial_state = {
                    "user_query": user_query,
                    "k": _pipeline_defaults.get("k", 2),
                    "max_retry_for_groundness_checking": _pipeline_defaults.get("max_retry_for_groundness_checking", 1),
                    "max_retry_for_answer_relevant_checking": _pipeline_defaults.get("max_retry_for_answer_relevant_checking", 1),
                    "max_turns_before_summarisation": _pipeline_defaults.get("max_turns_before_summarisation", 2),
                    "messages_to_include": _pipeline_defaults.get("messages_to_include", 0),
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            async for chunk in workflow.astream(
                initial_state,
                {"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                node_name = list(chunk.keys())[0]
                node_data = chunk[node_name]
                details = _extract_node_details(node_name, node_data)
                yield f"event: node_complete\ndata: {json.dumps({ 'node': node_name, 'details': details })}\n\n"
            response = await workflow.aget_state(
                    config={"configurable": {"thread_id": thread_id}}
                )
            ai_response = response.values["generated_response"]
            in_tokens = response.values.get("input_tokens", 0)
            out_tokens = response.values.get("output_tokens", 0)
            route = response.values.get("retrieval_required")
            contexts = response.values.get("relevant_contexts") or []
            web_searched = response.values.get("web_searched", False)
            is_grounded = response.values.get("is_grounded")
            is_answer_relevant = response.values.get("is_answer_relevant")

            yield f"event: done\ndata: {json.dumps({ 'response': ai_response, 'in_tokens': in_tokens, 'out_tokens': out_tokens, 'route': route, 'contexts': contexts, 'web_searched': web_searched, 'is_grounded': is_grounded, 'is_answer_relevant': is_answer_relevant })}\n\n"

    except Exception as e:
        print(e)
        yield f"event: error\ndata: {json.dumps({ 'message': str(e) })}\n\n"


@app.get("/rag/stream")
async def rag_stream(thread_id: str, query: str):
    return StreamingResponse(
        run_workflow(thread_id, query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
