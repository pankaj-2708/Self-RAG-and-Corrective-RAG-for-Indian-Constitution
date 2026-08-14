import { useCallback, useEffect, useRef, useState } from 'react'
import ThreadRail from './components/ThreadRail'
import Composer from './components/Composer'
import Exchange from './components/Exchange'
import Opening from './components/Opening'
import { fetchThreads, streamAnswer } from './lib/api'

function newThreadId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  return `thread-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function pairMessages(messages) {
  const turns = []
  let pending = null

  for (const message of messages) {
    if (message.role === 'user') {
      if (pending) turns.push(pending)
      pending = {
        id: `h${turns.length}`,
        question: message.content,
        answer: '',
        steps: [],
        status: 'done',
        contexts: [],
        inTokens: 0,
        outTokens: 0,
      }
    } else if (pending) {
      pending.answer = message.content
      turns.push(pending)
      pending = null
    }
  }

  if (pending) turns.push(pending)
  return turns
}

function threadTitle(thread) {
  const first = thread?.messages?.find((message) => message.role === 'user')
  if (!first?.content) return 'Fresh thread'
  return first.content.length > 72 ? `${first.content.slice(0, 72)}...` : first.content
}

export default function App() {
  const [threads, setThreads] = useState([])
  const [threadsLoading, setThreadsLoading] = useState(true)
  const [online, setOnline] = useState(true)
  const [activeId, setActiveId] = useState(() => newThreadId())
  const [turnsByThread, setTurnsByThread] = useState({})
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [railOpen, setRailOpen] = useState(true)
  const [railWidth, setRailWidth] = useState(() => {
    const saved = localStorage.getItem('railWidth')
    return saved ? Math.max(180, Math.min(450, Number(saved))) : 280
  })
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme')
    if (saved === 'dark' || saved === 'light') return saved
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  const isResizingRef = useRef(false)
  const abortRef = useRef(null)
  const scrollerRef = useRef(null)

  const turns = turnsByThread[activeId] ?? []
  const activeThread = threads.find((thread) => thread.id === activeId)
  const activeTurn = turns[turns.length - 1] ?? null

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('railWidth', railWidth)
  }, [railWidth])

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))
  }, [])

  const handleMouseDown = useCallback((event) => {
    event.preventDefault()
    isResizingRef.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const handleMouseMove = (moveEvent) => {
      if (!isResizingRef.current) return
      setRailWidth(Math.max(220, Math.min(420, moveEvent.clientX)))
    }

    const handleMouseUp = () => {
      if (!isResizingRef.current) return
      isResizingRef.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
  }, [])

  const loadThreads = useCallback(async () => {
    try {
      const list = await fetchThreads()
      setThreads(list)
      setOnline(true)
    } catch {
      setOnline(false)
    } finally {
      setThreadsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadThreads()
  }, [loadThreads])

  useEffect(() => {
    if (!turns.length) return
    const element = scrollerRef.current
    if (element) element.scrollTop = element.scrollHeight
  }, [turns.length, busy])

  const patchTurn = useCallback((threadId, turnId, patch) => {
    setTurnsByThread((prev) => ({
      ...prev,
      [threadId]: (prev[threadId] ?? []).map((turn) =>
        turn.id === turnId ? { ...turn, ...patch } : turn
      ),
    }))
  }, [])

  const ask = useCallback(
    async (question, threadId = activeId) => {
      const text = question.trim()
      if (!text || busy) return

      const turnId = `t${Date.now()}`
      setTurnsByThread((prev) => ({
        ...prev,
        [threadId]: [
          ...(prev[threadId] ?? []),
          {
            id: turnId,
            question: text,
            answer: '',
            steps: [],
            status: 'running',
            contexts: [],
            inTokens: 0,
            outTokens: 0,
          },
        ],
      }))
      setDraft('')
      setBusy(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        for await (const event of streamAnswer({ threadId, query: text, signal: controller.signal })) {
          if (event.type === 'step') {
            setTurnsByThread((prev) => ({
              ...prev,
              [threadId]: (prev[threadId] ?? []).map((turn) =>
                turn.id === turnId
                  ? {
                      ...turn,
                      steps: [...turn.steps, { node: event.node, details: event.details ?? {} }],
                    }
                  : turn
              ),
            }))
          } else if (event.type === 'answer') {
            patchTurn(threadId, turnId, {
              status: 'done',
              answer: event.text,
              contexts: event.contexts,
              inTokens: event.inTokens,
              outTokens: event.outTokens,
              route: event.route,
              webSearched: event.webSearched,
            })
          }
        }

        setOnline(true)
        loadThreads()
      } catch (error) {
        if (error.name === 'AbortError') {
          patchTurn(threadId, turnId, { status: 'done', answer: '', error: 'You stopped this answer.' })
        } else {
          patchTurn(threadId, turnId, { status: 'error', error: error.message })
          setOnline(false)
        }
      } finally {
        abortRef.current = null
        setBusy(false)
      }
    },
    [activeId, busy, loadThreads, patchTurn]
  )

  const selectThread = useCallback(
    (id) => {
      const thread = threads.find((item) => item.id === id)
      setActiveId(id)
      setTurnsByThread((prev) =>
        prev[id]?.length ? prev : { ...prev, [id]: pairMessages(thread?.messages ?? []) }
      )
    },
    [threads]
  )

  const startThread = useCallback(() => {
    setActiveId(newThreadId())
    setDraft('')
  }, [])

  const stop = useCallback(() => abortRef.current?.abort(), [])

  return (
    <div className={`app${railOpen ? ' app--sidebar-open' : ''}`}>
      <ThreadRail
        threads={threads}
        activeId={activeId}
        onSelect={selectThread}
        onNew={startThread}
        open={railOpen}
        onToggle={() => setRailOpen((prev) => !prev)}
        online={online}
        loading={threadsLoading}
        width={railWidth}
      />

      {railOpen ? (
        <div className="rail-resizer" onMouseDown={handleMouseDown} title="Drag to resize sidebar" />
      ) : null}

      <main className="main">
        <div className="main__ambient" aria-hidden="true">
          <div className="main__ambient-orb main__ambient-orb--one" />
          <div className="main__ambient-orb main__ambient-orb--two" />
        </div>

        <div className="topbar">
          <div className="topbar__left">
            <button
              className="topbar__toggle"
              onClick={() => setRailOpen((prev) => !prev)}
              aria-label={railOpen ? 'Close sidebar' : 'Open sidebar'}
              title={railOpen ? 'Close sidebar' : 'Open sidebar'}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <line x1="9" y1="3" x2="9" y2="21" />
              </svg>
            </button>

            <div className="topbar__identity">
              <span className="topbar__mark">Samvidhan</span>
              <span className="topbar__submark">
                {turns.length ? threadTitle(activeThread) : 'Constitution and IPC reasoning assistant'}
              </span>
            </div>
          </div>

          <div className="topbar__right">
            <div className={`topbar__status ${busy ? 'topbar__status--busy' : ''}`}>
              <span className={`dot ${online ? 'dot--live' : 'dot--down'}`} />
              <span>{online ? (busy ? 'Workflow running' : 'Graph ready') : 'Graph offline'}</span>
              {activeTurn?.steps?.length ? (
                <span className="topbar__status-meta">{activeTurn.steps.length} live updates</span>
              ) : null}
            </div>

            <button
              className="topbar__theme-btn"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                  </svg>
                  <span>Dark mode</span>
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="5" />
                    <line x1="12" y1="1" x2="12" y2="3" />
                    <line x1="12" y1="21" x2="12" y2="23" />
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                    <line x1="1" y1="12" x2="3" y2="12" />
                    <line x1="21" y1="12" x2="23" y2="12" />
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                  </svg>
                  <span>Light mode</span>
                </>
              )}
            </button>
          </div>
        </div>

        <div className={`scroller${turns.length ? '' : ' scroller--opening'}`} ref={scrollerRef}>
          {turns.length === 0 ? (
            <Opening onPick={(text) => ask(text)} />
          ) : (
            <div className="sheet">
              {turns.map((turn) => (
                <Exchange key={turn.id} turn={turn} onRetry={(item) => ask(item.question)} />
              ))}
            </div>
          )}
        </div>

        <Composer
          value={draft}
          onChange={setDraft}
          onSubmit={() => ask(draft)}
          onStop={stop}
          busy={busy}
        />
      </main>
    </div>
  )
}
