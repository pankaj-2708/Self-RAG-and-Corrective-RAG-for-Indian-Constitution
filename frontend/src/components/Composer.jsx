import { useEffect, useRef } from 'react'

export default function Composer({ value, onChange, onSubmit, onStop, busy }) {
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [value])

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (!busy && value.trim()) onSubmit()
    }
  }

  return (
    <div className="composer">
      <div className="composer__inner">
        <div className="composer__mark">Ask</div>
        <div className="composer__field">
          <textarea
            ref={ref}
            className="composer__input"
            rows={1}
            value={value}
            placeholder="Ask about the Constitution or the IPC"
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={handleKeyDown}
            aria-label="Your question"
          />
          {busy ? (
            <button className="composer__send composer__send--stop" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button className="composer__send" onClick={onSubmit} disabled={!value.trim()}>
              Ask
              <svg width="11" height="10" viewBox="0 0 11 10" aria-hidden="true">
                <path
                  d="M1 5h8M6 1.5L9.5 5 6 8.5"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
