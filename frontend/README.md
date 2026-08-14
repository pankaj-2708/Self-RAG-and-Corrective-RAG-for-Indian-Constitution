# Samvidhan — web client

Frontend for the Self-RAG / Corrective-RAG graph over the Constitution of India
and the Indian Penal Code.

The thing this interface is built around: the graph does not just answer, it
**grades its own answer and redoes the work when a check fails**. That process is
normally invisible. Here it is drawn in the margin beside every answer, live, as
the nodes complete — passes in viridian, self-caught corrections in oxblood.

## Run it

```bash
# 1 — backend (from the repo root)
cd src/Backend
uvicorn main:app --reload --port 8000

# 2 — frontend (from this folder)
npm install
npm run dev          # http://localhost:5173
```

In development the app calls `/api/*`, which Vite proxies to
`http://127.0.0.1:8000`, so CORS never comes up. To point at a deployed backend
instead, copy `.env.example` to `.env` and set `VITE_API_BASE`.

## Apply the backend patch

`backend/main.py` in this folder is a drop-in replacement for
`src/Backend/main.py`. Copy it over:

```bash
cp backend/main.py ../src/Backend/main.py
```

It changes three things:

| | Before | After |
|---|---|---|
| SSE format | `event: done\n {json}` — no `data:` field, no blank-line terminator, so spec-compliant clients drop the final answer | proper `event: … \ndata: … \n\n` |
| `done` payload | answer + token counts | also `contexts`, `route`, `web_searched`, `is_grounded`, `is_answer_relevant` |
| health | none | `GET /health` |

The app works **without** the patch — the SSE reader in `src/lib/sse.js`
tolerates the malformed shape. What you lose is the passage list under each
answer and the route line, since those fields aren't sent.

## API surface consumed

### `GET /all_chats`

```json
{
  "status": "success",
  "response": [
    { "chat_id": "…", "messages": [{ "role": "human", "content": "…" }] }
  ]
}
```

`role` is a LangChain message type (`human` / `ai`); the client maps these to
`user` / `assistant` and drops everything else.

### `GET /rag/stream?thread_id={uuid}&query={text}`

`text/event-stream`. `thread_id` is generated client-side with
`crypto.randomUUID()` — the backend creates the checkpoint on first use, so no
"create thread" call is needed.

```
event: node_complete
data: {"node": "retrieval_decider_node"}

event: done
data: {"response": "…", "in_tokens": 0, "out_tokens": 0,
       "contexts": ["…"], "route": "retrieval", "web_searched": false,
       "is_grounded": "fully_supported", "is_answer_relevant": true}
```

On failure the graph emits `event: error` with `{"message": "…"}`.

`retrieve_node` and `is_relevant_node` fan out and therefore emit once per
query and once per passage. The client folds consecutive repeats into a count
(`Searched the corpus ×3`) rather than listing them individually.

## Layout

```
src/
  lib/
    sse.js        forgiving event-stream reader (handles both SSE shapes)
    api.js        the two endpoints, typed into plain objects
    nodes.js      graph node names → reader-facing labels + pass/correction kind
    markdown.js   small markdown → React renderer, no HTML constructed
  components/
    ThreadRail    saved threads, connection state
    Opening       empty state — a page of the statute
    Exchange      one question and its answer
    MarginLedger  the live self-verification record
    Passages      the passages the answer was built from
    Composer      the input
  App.jsx         state, streaming, thread switching
  styles.css      design tokens and layout
```

## Design notes

**Palette.** Taken from a printed statute: cool bond paper (`#ECEEE7`), iron-gall
ink (`#151E2A`), and the two inks a reader marks a page with — oxblood
(`#8E2B22`) for a correction, viridian (`#2E5C4F`) for something that holds up.
Brass (`#9C7C3C`) appears once, on the seal.

**Type.** Newsreader for the statute text and questions, its italic a nod to the
handwritten italic of the original engrossed Constitution. Archivo for the
interface. IBM Plex Mono for the ledger and counts. Tiro Devanagari Hindi for
the wordmark.

**Structure.** The marginal note is a real device in the Constitution — the
short heading that sits beside each article. Here the margin holds the
verification record, which is the one thing about this system worth looking at.

Responsive to 380px, keyboard-focusable throughout, `prefers-reduced-motion`
respected.

## Not covered

There is no auth, and no thread deletion — the backend exposes neither.
