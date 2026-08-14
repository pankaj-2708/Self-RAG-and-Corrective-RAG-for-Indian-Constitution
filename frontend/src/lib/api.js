import { readEventStream } from './sse'

/**
 * Where the FastAPI server lives.
 * Falls back to 'http://127.0.0.1:8000' if VITE_API_BASE environment variable is not present.
 */
const BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export const apiOrigin = BASE

class ApiError extends Error {
  constructor(message, { cause, status } = {}) {
    super(message)
    this.name = 'ApiError'
    this.cause = cause
    this.status = status
  }
}

const OFFLINE = `Can't reach the API at ${apiOrigin}. Start the backend with \`uvicorn main:app --reload\` from src/Backend, then try again.`

/**
 * GET /all_chats — every saved thread with its message history.
 * @returns {Promise<Array<{id: string, messages: Array<{role: string, content: string}>}>>}
 */
export async function fetchThreads() {
  let response
  try {
    response = await fetch(`${BASE}/all_chats`)
  } catch (cause) {
    throw new ApiError(OFFLINE, { cause })
  }

  if (!response.ok) {
    throw new ApiError(`The server returned ${response.status} for /all_chats.`, {
      status: response.status,
    })
  }

  const body = await response.json()
  if (body.status !== 'success') {
    throw new ApiError(body.error || 'The server could not load saved threads.')
  }

  return (body.response || []).map((thread) => ({
    id: thread.chat_id,
    messages: (thread.messages || [])
      .filter((m) => m.role === 'human' || m.role === 'ai')
      .map((m) => ({
        role: m.role === 'human' ? 'user' : 'assistant',
        content: m.content,
      })),
  }))
}

/**
 * GET /rag/stream — run one turn of the graph, streaming node completions.
 *
 * Yields, in order:
 *   { type: 'step',  node: string }
 *   { type: 'answer', text, inTokens, outTokens, contexts, route, webSearched, grounded, relevant }
 *
 * Throws an ApiError if the graph reports a failure.
 *
 * @param {{threadId: string, query: string, signal?: AbortSignal}} args
 */
export async function* streamAnswer({ threadId, query, signal }) {
  const url = `${BASE}/rag/stream?thread_id=${encodeURIComponent(
    threadId
  )}&query=${encodeURIComponent(query)}`

  let response
  try {
    response = await fetch(url, { signal, headers: { Accept: 'text/event-stream' } })
  } catch (cause) {
    if (cause.name === 'AbortError') throw cause
    throw new ApiError(OFFLINE, { cause })
  }

  if (!response.ok) {
    throw new ApiError(`The server returned ${response.status} while answering.`, {
      status: response.status,
    })
  }

  let delivered = false

  for await (const { event, data } of readEventStream(response)) {
    if (event === 'node_complete' && data?.node) {
      yield { type: 'step', node: data.node, details: data.details ?? {} }
      continue
    }

    if (event === 'error') {
      throw new ApiError(data?.message || 'The graph stopped with an error.')
    }

    if (event === 'done') {
      delivered = true
      yield {
        type: 'answer',
        text: data?.response ?? '',
        inTokens: data?.in_tokens ?? 0,
        outTokens: data?.out_tokens ?? 0,
        // The fields below only arrive once backend/main.py is applied.
        contexts: data?.contexts ?? [],
        route: data?.route ?? null,
        webSearched: data?.web_searched ?? false,
        grounded: data?.is_grounded ?? null,
        relevant: data?.is_answer_relevant ?? null,
      }
    }
  }

  if (!delivered) {
    throw new ApiError(
      'The stream closed before an answer arrived. Check the backend console for a traceback.'
    )
  }
}

export { ApiError }
