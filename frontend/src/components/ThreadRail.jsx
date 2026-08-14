function titleOf(thread) {
  const first = thread.messages.find((message) => message.role === 'user')
  if (!first) return 'Empty thread'
  return first.content.length > 68 ? `${first.content.slice(0, 68)}...` : first.content
}

export default function ThreadRail({
  threads,
  activeId,
  onSelect,
  onNew,
  open,
  onToggle,
  online,
  loading,
  width,
}) {
  if (!open) {
    return (
      <aside className="rail rail--collapsed" aria-label="Sidebar collapsed">
        <button
          className="rail__collapse-btn"
          onClick={onToggle}
          aria-label="Open sidebar"
          title="Open sidebar"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <line x1="9" y1="3" x2="9" y2="21" />
          </svg>
        </button>
      </aside>
    )
  }

  return (
    <aside className={`rail${open ? ' rail--open' : ''}`} style={width ? { width: `${width}px` } : undefined}>
      <div className="rail__head">
        <div className="rail__head-top">
          <div>
            <span className="rail__mark">Samvidhan</span>
            <span className="rail__wordmark">Constitution and IPC assistant</span>
          </div>
          <button className="rail__close-btn" onClick={onToggle} aria-label="Close sidebar" title="Close sidebar">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
              <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z" />
            </svg>
          </button>
        </div>
        <p className="rail__tagline">
          Verified answers with retrieval, grading, and revision traces exposed while the graph works.
        </p>
      </div>

      <div className="rail__summary">
        <div className="rail__summary-card">
          <span className="rail__summary-label">Saved</span>
          <strong>{threads.length}</strong>
        </div>
        <div className="rail__summary-card">
          <span className="rail__summary-label">Status</span>
          <strong>{online ? 'Live' : 'Offline'}</strong>
        </div>
      </div>

      <button className="rail__new" onClick={onNew}>
        <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">
          <path d="M5.5 1v9M1 5.5h9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
        Start a new thread
      </button>

      <p className="rail__label">Saved threads</p>

      <ul className="rail__list">
        {loading ? <li className="rail__empty">Loading...</li> : null}
        {!loading && !threads.length ? (
          <li className="rail__empty">Nothing saved yet. Your first question starts a thread.</li>
        ) : null}
        {threads.map((thread) => {
          const questionCount = thread.messages.filter((message) => message.role === 'user').length

          return (
            <li key={thread.id}>
              <button className="rail__item" aria-current={thread.id === activeId} onClick={() => onSelect(thread.id)}>
                <span className="rail__item-title">{titleOf(thread)}</span>
                <span className="rail__count">
                  {questionCount} question{questionCount === 1 ? '' : 's'}
                </span>
              </button>
            </li>
          )
        })}
      </ul>

      <div className="rail__foot">
        <span className={`dot ${online ? 'dot--live' : 'dot--down'}`} />
        {online ? 'Graph connected' : 'Graph unreachable'}
      </div>
    </aside>
  )
}
