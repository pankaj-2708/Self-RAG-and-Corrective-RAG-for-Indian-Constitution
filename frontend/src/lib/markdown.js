/**
 * Render the model's markdown answer.
 *
 * The model emits light markdown — headings, lists, bold, inline code. We pass
 * the raw text straight to `marked`, which handles the standard formatting
 * (including ordered-list numbering) natively. No custom numbering, CSS
 * counters, or index overrides are applied here; whatever the model wrote is
 * what the browser renders.
 */
import { marked } from 'marked'

marked.setOptions({
  gfm: true,
  breaks: false,
})

/**
 * @param {string} source
 * @returns {string} the HTML produced by `marked` for `source`
 */
export function renderMarkdown(source) {
  return marked.parse(String(source ?? ''))
}
