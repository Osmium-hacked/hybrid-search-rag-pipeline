const CITATION = /[[【]\s*(?:Source:\s*)?(chunk_\d+(?:\s*,\s*(?:Source:\s*)?chunk_\d+)*)\s*[\]】]/gi

function citedIds(match: string) {
  return match.match(/chunk_\d+/gi) ?? []
}

/** Rewrites `[Source: chunk_14]` into Markdown links that the answer renders as numbered chips. */
export function linkCitations(answer: string, sourceIds: string[]) {
  return answer.replace(CITATION, (_, ids: string) =>
    citedIds(ids)
      .map((id) => `[${sourceIds.indexOf(id) + 1 || '?'}](#cite-${id})`)
      .join(' '),
  )
}

/** Converts \( \) and \[ \] math delimiters, which models often use, into the $ form remark-math parses. */
export function normalizeMath(answer: string) {
  return answer
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, math: string) => `\n$$\n${math.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, math: string) => `$${math.trim()}$`)
}

export function citedChunks(answer: string) {
  const cited = new Set<string>()
  for (const [, ids] of answer.matchAll(CITATION)) {
    citedIds(ids).forEach((id) => cited.add(id))
  }
  return cited
}

export function formatDuration(ms: number) {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`
}

export function shortModel(model: string) {
  return model.split('/').pop() ?? model
}

export function formatMegabytes(bytes: number) {
  return `${Math.round(bytes / (1024 * 1024))} MB`
}

export function passages(count: number) {
  return `${count} ${count === 1 ? 'passage' : 'passages'}`
}

/** Where a passage came from inside its file, e.g. "Page 3" or "Slide 2". */
export function sourceLocation(source: { page: number | null; location?: string | null }) {
  return source.location ?? (source.page ? `Page ${source.page}` : null)
}
