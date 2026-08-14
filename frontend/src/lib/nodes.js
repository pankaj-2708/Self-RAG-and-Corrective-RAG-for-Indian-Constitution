/**
 * Graph node names, translated into something a reader can follow.
 *
 * `kind` drives the colour of the mark in the margin ledger:
 *   read       — the system is gathering material
 *   check      — the system is grading its own work
 *   correction — a check failed and the answer is being redone (oxblood)
 *   record     — bookkeeping after the answer is settled
 */
export const NODES = {
  retrieval_decider_node: { label: 'Chose where to look', kind: 'read' },
  generate_retriever_query_node: { label: 'Drafted corpus queries', kind: 'read' },
  retrieve_node: { label: 'Searched the corpus', kind: 'read' },
  aggregate_retrieval: { label: 'Collected passages', kind: 'read' },
  generate_web_search_query_node: { label: 'Drafted web queries', kind: 'read' },
  web_search_node: { label: 'Searched the web', kind: 'read' },
  is_relevant_node: { label: 'Graded a passage', kind: 'check' },
  aggregate_relevance: { label: 'Kept the relevant passages', kind: 'check' },
  answer_from_context_node: { label: 'Drafted the answer', kind: 'read' },
  direct_generation_node: { label: 'Answered without retrieval', kind: 'read' },
  check_answer_grounded_node: { label: 'Checked every claim against the passages', kind: 'check' },
  revise_answer_node: { label: 'Revised — claims were not fully supported', kind: 'correction' },
  is_answer_relevant_node: { label: 'Checked the answer addresses the question', kind: 'check' },
  rewrite_answer_node: { label: 'Rewrote — the answer missed the question', kind: 'correction' },
  memory_node: { label: 'Filed the exchange', kind: 'record' },
  modify_short_term_memory_node: { label: 'Summarised the thread', kind: 'record' },
}

export function describeNode(node) {
  return NODES[node] ?? { label: node.replace(/_node$/, '').replace(/_/g, ' '), kind: 'read' }
}

/**
 * Fold a raw sequence of node names into ledger entries, collapsing the
 * fan-out nodes (which fire once per query or per passage) into a count.
 */
export function toLedger(nodeSequence) {
  const entries = []
  for (const step of nodeSequence) {
    const node = typeof step === 'string' ? step : step.node
    const details = typeof step === 'string' ? {} : (step.details ?? {})
    const last = entries[entries.length - 1]
    if (last && last.node === node) {
      last.count += 1
      last.details = details
      last.allDetails = [...(last.allDetails || []), details]
    } else {
      entries.push({ node, count: 1, ...describeNode(node), details, allDetails: [details] })
    }
  }
  return entries
}

/** One-line verdict for the collapsed stamp. */
export function summariseLedger(entries) {
  const steps = entries.reduce((total, entry) => total + entry.count, 0)
  const corrections = entries
    .filter((entry) => entry.kind === 'correction')
    .reduce((total, entry) => total + entry.count, 0)
  return { steps, corrections }
}
