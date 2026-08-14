import { useEffect, useState } from 'react'
import { toLedger, summariseLedger } from '../lib/nodes'

const ROUTE_LABELS = {
  retrieval: 'Searching the local corpus',
  web_search: 'Searching the web',
  None: 'Answering directly',
}

function latestStatus(entry) {
  if (!entry) return 'Starting workflow'

  switch (entry.node) {
    case 'retrieval_decider_node':
      return ROUTE_LABELS[entry.details?.route] || 'Choosing a route'
    case 'generate_retriever_query_node':
      return 'Drafting corpus queries'
    case 'retrieve_node':
      return `Retrieved ${entry.details?.retrieved_count ?? 0} passages`
    case 'aggregate_relevance':
      return 'Filtering the most relevant evidence'
    case 'generate_web_search_query_node':
      return 'Drafting web queries'
    case 'web_search_node':
      return `Collected ${entry.details?.web_result_count ?? 0} web results`
    case 'answer_from_context_node':
    case 'direct_generation_node':
      return 'Writing the answer'
    case 'check_answer_grounded_node':
      return entry.details?.is_grounded === 'fully_supported'
        ? 'Checking support for every claim'
        : 'Revising unsupported claims'
    case 'is_answer_relevant_node':
      return entry.details?.is_relevant ? 'Checking relevance to the question' : 'Rewriting for relevance'
    case 'memory_node':
    case 'modify_short_term_memory_node':
      return 'Saving this exchange'
    default:
      return entry.label
  }
}

function liveDetails(entries) {
  const route = entries.find((entry) => entry.node === 'retrieval_decider_node')?.details?.route
  const retrieverQueries = entries.find((entry) => entry.node === 'generate_retriever_query_node')?.details?.retriever_queries
  const webQueries = entries.find((entry) => entry.node === 'generate_web_search_query_node')?.details?.web_queries
  const webTitles = entries.find((entry) => entry.node === 'web_search_node')?.details?.web_titles
  const retrieveEntry = entries.find((entry) => entry.node === 'retrieve_node')
  const relevanceEntry = entries.find((entry) => entry.node === 'is_relevant_node')

  const items = []

  if (route) {
    items.push({ label: 'Route', value: ROUTE_LABELS[route] || route })
  }

  if (retrieveEntry?.details) {
    const titles = retrieveEntry.details.retrieved_titles ?? []
    const previews = retrieveEntry.details.retrieved_previews ?? []
    const rows = previews.map((preview, index) => ({
      index,
      title: titles[index] || `Passage ${index + 1}`,
      preview,
    }))
    if (rows.length) {
      items.push({ label: 'Retrieved', value: rows, kind: 'passages' })
    }
  }

  if (relevanceEntry?.allDetails?.length) {
    const keptIndices = []
    for (const detail of relevanceEntry.allDetails) {
      if (Array.isArray(detail?.kept_indices)) {
        keptIndices.push(...detail.kept_indices)
      }
    }
    const uniqueKept = [...new Set(keptIndices)].sort((a, b) => a - b)
    const titles = retrieveEntry?.details?.retrieved_titles ?? []
    const previews = retrieveEntry?.details?.retrieved_previews ?? []
    const keptRows = uniqueKept.map((index) => ({
      index,
      title: titles[index] || `Passage ${index + 1}`,
      preview: previews[index] || '',
    }))
    if (keptRows.length) {
      items.push({ label: 'Kept', value: keptRows, kind: 'passages' })
    }
  }

  if (retrieverQueries?.length) {
    items.push({ label: 'Queries', value: retrieverQueries.map((item) => item.query).slice(0, 3) })
  }

  if (webQueries?.length) {
    items.push({ label: 'Web queries', value: webQueries.slice(0, 3) })
  }

  if (webTitles?.length) {
    items.push({ label: 'Sources', value: webTitles.slice(0, 3) })
  }

  return items
}

function PassageRow({ row }) {
  return (
    <div className="progress-card__passage">
      <span className="progress-card__passage-index">#{row.index + 1}</span>
      <div className="progress-card__passage-body">
        <div className="progress-card__passage-title">{row.title}</div>
        {row.preview ? <div className="progress-card__passage-preview">{row.preview}</div> : null}
      </div>
    </div>
  )
}

function DetailValue({ value, kind }) {
  if (kind === 'passages' && Array.isArray(value)) {
    return (
      <div className="progress-card__passages">
        {value.map((row) => (
          <PassageRow key={`p-${row.index}`} row={row} />
        ))}
      </div>
    )
  }

  if (Array.isArray(value)) {
    return (
      <div className="progress-card__list">
        {value.map((item, index) => (
          <div key={`${item}-${index}`} className="progress-card__item">
            {item}
          </div>
        ))}
      </div>
    )
  }

  return <div className="progress-card__value">{value}</div>
}

export default function MarginLedger({ steps, running }) {
  const [open, setOpen] = useState(false)
  const entries = toLedger(steps)
  const live = liveDetails(entries)
  const latestEntry = entries[entries.length - 1] ?? null
  const { steps: total, corrections } = summariseLedger(entries)

  useEffect(() => {
    if (!running) setOpen(false)
  }, [running])

  if (!entries.length && !running) return null

  return (
    <section className={`progress${running || open ? ' progress--expanded' : ''}`}>
      <div className={`progress__summary ${running ? 'progress__summary--running' : ''}`}>
        <div className="progress__pulse" aria-hidden="true" />
        <div className="progress__copy">
          <div className="progress__label">{running ? 'Working' : 'Completed'}</div>
          <div className="progress__title">{latestStatus(latestEntry)}</div>
        </div>
        <div className="progress__meta">
          <span>{total} step{total === 1 ? '' : 's'}</span>
          {corrections ? <span>{corrections} revision{corrections === 1 ? '' : 's'}</span> : null}
        </div>
      </div>

      {(running || open) && live.length ? (
        <div className="progress__details">
          {live.map((item) => (
            <div key={item.label} className="progress-card">
              <div className="progress-card__label">{item.label}</div>
              <DetailValue value={item.value} kind={item.kind} />
            </div>
          ))}
        </div>
      ) : null}

      <div className="progress__history">
        <button className="progress__toggle" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
          <span>{open || running ? 'Hide progress details' : 'Show progress details'}</span>
        </button>

        {open || running ? (
          <ol className="progress-steps">
            {entries.map((entry, index) => {
              const liveEntry = running && index === entries.length - 1
              return (
                <li
                  key={`${entry.node}-${index}`}
                  className={`progress-step progress-step--${entry.kind}${liveEntry ? ' progress-step--live' : ''}`}
                >
                  <span className="progress-step__title">
                    {entry.label}
                    {entry.count > 1 ? <span className="progress-step__count"> x{entry.count}</span> : null}
                  </span>
                </li>
              )
            })}
          </ol>
        ) : null}
      </div>
    </section>
  )
}
