from fastapi import FastAPI
import sys
import os
import json
import asyncio
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import messages_to_dict

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)
from workflow import get_workflow

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
    """
    try:
        results = []

        async with get_workflow() as (workflow, ck_ptr):
            # Query the underlying SQLite DB directly for all distinct thread_ids
            async with ck_ptr.conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
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
                    "k": 2,
                    "max_retry_for_groundness_checking": 1,
                    "max_retry_for_answer_relevant_checking": 1,
                }
            else:
                initial_state = {
                    "user_query": user_query,
                    "k": 2,
                    "max_retry_for_groundness_checking": 1,
                    "max_retry_for_answer_relevant_checking": 1,
                    "max_turns_before_summarisation": 2,
                    "messages_to_include": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            async for chunk in workflow.astream(
                initial_state,
                {"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                node_name = list(chunk.keys())[0]
                yield f"event: node_complete\ndata: {json.dumps({
                                "node": f"{node_name}",
                            })} \n\n"
            response = await workflow.aget_state(
                    config={"configurable": {"thread_id": thread_id}}
                )
            ai_response = response.values["generated_response"]
            in_tokens = response.values.get("input_tokens", 0)
            out_tokens = response.values.get("output_tokens", 0)

            yield f"event: done\n {json.dumps({"response": ai_response,"in_tokens":in_tokens,"out_tokens":out_tokens})}"

    except Exception as e:
        print(e)
        yield f"event:error \n {json.dumps({"message": str(e)})}"


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
    uvicorn.run(app)
