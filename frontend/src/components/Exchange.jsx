import MarginLedger from './MarginLedger'
import Passages from './Passages'
import { renderMarkdown } from '../lib/markdown'

const ROUTE_WORDS = {
  retrieval: 'Used the local corpus',
  web_search: 'Used the web',
  None: 'Answered directly',
}

function Meter({ turn }) {
  const bits = []
  if (turn.route && ROUTE_WORDS[turn.route]) bits.push(ROUTE_WORDS[turn.route])
  if (turn.webSearched && turn.route === 'retrieval') bits.push('fell back to web search')
  if (turn.inTokens || turn.outTokens) {
    bits.push(`${turn.inTokens.toLocaleString()} input`)
    bits.push(`${turn.outTokens.toLocaleString()} output`)
  }
  if (!bits.length) return null

  return (
    <div className="meter">
      {bits.map((bit, index) => (
        <span key={index} className="meter__chip">
          {bit}
        </span>
      ))}
    </div>
  )
}

function AnswerSkeleton() {
  return (
    <div className="answer-skeleton" aria-hidden="true">
      <span className="answer-skeleton__line answer-skeleton__line--wide" />
      <span className="answer-skeleton__line" />
      <span className="answer-skeleton__line answer-skeleton__line--mid" />
    </div>
  )
}

export default function Exchange({ turn, onRetry }) {
  const running = turn.status === 'running'

  return (
    <article className="exchange">
      <div className="exchange__question">
        <div className="question-bubble">{turn.question}</div>
      </div>

      <div className="exchange__answer">
        {running && !turn.answer ? (
          <>
            <p className="thinking">Working on it</p>
            <AnswerSkeleton />
          </>
        ) : null}

        {turn.answer ? (
          <div className="answer" dangerouslySetInnerHTML={{ __html: renderMarkdown(turn.answer) }} />
        ) : null}

        {turn.error ? (
          <div className="notice" role="alert">
            {turn.error}
            {onRetry ? (
              <div>
                <button className="notice__retry" onClick={() => onRetry(turn)}>
                  Ask again
                </button>
              </div>
            ) : null}
          </div>
        ) : null}

        <MarginLedger steps={turn.steps} running={running} />

        {turn.status === 'done' ? (
          <>
            <Passages contexts={turn.contexts} />
            <Meter turn={turn} />
          </>
        ) : null}
      </div>
    </article>
  )
}
