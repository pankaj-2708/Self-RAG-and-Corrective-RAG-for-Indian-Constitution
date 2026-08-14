import { useState } from 'react'

function Passage({ text, index }) {
  const [expanded, setExpanded] = useState(false)
  const long = text.length > 320

  return (
    <li className="passage">
      <span className="passage__index">{String(index + 1).padStart(2, '0')}</span>
      <div>
        <p className={`passage__text${long && !expanded ? ' passage__text--clamped' : ''}`}>{text}</p>
        {long && (
          <button className="passage__more" onClick={() => setExpanded((v) => !v)}>
            {expanded ? 'Show less' : 'Show the full passage'}
          </button>
        )}
      </div>
    </li>
  )
}

/**
 * The passages the answer was actually built from — the ones that survived
 * relevance grading, not everything that came back from the retriever.
 */
export default function Passages({ contexts }) {
  const [open, setOpen] = useState(false)
  if (!contexts?.length) return null

  return (
    <div className="passages">
      <button className="passages__toggle" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <svg
          className={`passages__chevron${open ? ' passages__chevron--open' : ''}`}
          width="8"
          height="10"
          viewBox="0 0 8 10"
          aria-hidden="true"
        >
          <path d="M1 1l5 4-5 4" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" />
        </svg>
        {contexts.length} {contexts.length === 1 ? 'passage' : 'passages'} behind this answer
      </button>
      {open && (
        <ul className="passages__list">
          {contexts.map((text, index) => (
            <Passage key={index} text={text} index={index} />
          ))}
        </ul>
      )}
    </div>
  )
}
