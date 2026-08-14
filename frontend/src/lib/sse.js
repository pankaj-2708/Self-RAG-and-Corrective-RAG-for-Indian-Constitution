/**
 * A forgiving server-sent-events reader.
 *
 * The reference backend emits two shapes on the same stream:
 *
 *   1. Well-formed:  "event: node_complete\ndata: {...} \n\n"
 *   2. Malformed:    "event: done\n {...}"        <- no `data:`, no blank line
 *
 * A browser `EventSource` silently drops shape 2, which is where the final
 * answer lives. This reader accepts both, so the app keeps working whether or
 * not the backend patch in ../../backend/main.py has been applied.
 */

/** Pull the JSON payload out of one raw event record. */
function payloadOf(record) {
  const dataLines = []
  for (const line of record.split('\n')) {
    const trimmed = line.trimStart()
    if (trimmed.startsWith('data:')) dataLines.push(trimmed.slice(5).trim())
  }

  let raw = dataLines.join('\n').trim()

  // Fallback for records that omit `data:` entirely: take the outermost
  // brace-delimited span.
  if (!raw) {
    const start = record.indexOf('{')
    const end = record.lastIndexOf('}')
    if (start === -1 || end <= start) return null
    raw = record.slice(start, end + 1)
  }

  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

/** Pull the event name out of one raw event record. Defaults to `message`. */
function nameOf(record) {
  const match = record.match(/^\s*event:\s*([^\n{]+)/)
  return match ? match[1].trim() : 'message'
}

function toEvent(record) {
  if (!record.trim()) return null
  return { event: nameOf(record), data: payloadOf(record) }
}

/**
 * Read a `fetch` response body as a stream of `{ event, data }` objects.
 * @param {Response} response
 * @returns {AsyncGenerator<{event: string, data: object|null}>}
 */
export async function* readEventStream(response) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      let split
      while ((split = buffer.indexOf('\n\n')) !== -1) {
        const record = buffer.slice(0, split)
        buffer = buffer.slice(split + 2)
        const parsed = toEvent(record)
        if (parsed) yield parsed
      }
    }

    // Whatever is left when the connection closes is a final, unterminated
    // record — this is where the malformed `done` event lands.
    buffer += decoder.decode()
    const tail = toEvent(buffer)
    if (tail) yield tail
  } finally {
    reader.releaseLock()
  }
}
