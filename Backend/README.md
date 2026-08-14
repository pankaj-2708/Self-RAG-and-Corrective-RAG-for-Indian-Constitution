# Backend

FastAPI server that exposes the Self-RAG LangGraph workflow as a streaming HTTP API consumed by the React frontend.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/rag/stream` | Stream the RAG pipeline response as **Server-Sent Events (SSE)** |
| `GET` | `/all_chats` | Return all persisted chat threads with their messages |

### `GET /rag/stream`

Runs the full LangGraph pipeline for a given query and streams node-completion events back to the client.

**Query params:**

| Param | Type | Description |
|---|---|---|
| `thread_id` | `string` | Conversation thread ID (UUID). Determines which SQLite checkpoint to resume. |
| `query` | `string` | The user's question. |

**SSE events emitted:**

| Event | Payload | Description |
|---|---|---|
| `node_complete` | `{"node": "<node_name>"}` | Fired after each LangGraph node completes — used by the frontend to show live progress. |
| `done` | `{"response": "...", "in_tokens": N, "out_tokens": N}` | Final answer with token usage. |
| `error` | `{"message": "..."}` | Emitted if the pipeline throws an exception. |

### `GET /all_chats`

Returns all distinct conversation threads stored in the SQLite checkpoint database, including their full message history.

**Response:**
```json
{
  "status": "success",
  "response": [
    {
      "chat_id": "<thread_id>",
      "messages": [
        {"role": "human", "content": "..."},
        {"role": "ai", "content": "..."}
      ]
    }
  ]
}
```

## Running

From the project root:

```bash
uvicorn Backend.main:app --reload
```

The server starts at `http://localhost:8000`. CORS is open to all origins for local development.

## Files

| File | Description |
|---|---|
| [`main.py`](main.py) | FastAPI app — endpoint definitions, SSE streaming logic, workflow invocation |
