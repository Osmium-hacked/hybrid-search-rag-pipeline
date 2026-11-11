import { useEffect, useRef, useState } from 'react'
import type { Source } from '../api'
import { sourceLocation } from '../format'

interface EvidencePanelProps {
  sources: Source[]
  cited: Set<string>
  focused: { id: string; nonce: number } | null
}

export function EvidencePanel({ sources, cited, focused }: EvidencePanelProps) {
  const citedCount = sources.filter((source) => cited.has(source.chunk_id)).length

  return (
    <aside className="evidence" aria-label="Retrieved passages">
      <header>
        <h3>Evidence</h3>
        <span className="mono">
          {citedCount} of {sources.length} cited
        </span>
      </header>
      <ol>
        {sources.map((source, index) => (
          <EvidenceCard
            key={source.chunk_id}
            source={source}
            number={index + 1}
            isCited={cited.has(source.chunk_id)}
            focusNonce={focused?.id === source.chunk_id ? focused.nonce : null}
          />
        ))}
      </ol>
    </aside>
  )
}

interface EvidenceCardProps {
  source: Source
  number: number
  isCited: boolean
  focusNonce: number | null
}

function EvidenceCard({ source, number, isCited, focusNonce }: EvidenceCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [seenNonce, setSeenNonce] = useState(focusNonce)
  const card = useRef<HTMLLIElement>(null)

  // Clicking a citation opens its passage. Adjusted during render, as React recommends.
  if (focusNonce !== seenNonce) {
    setSeenNonce(focusNonce)
    if (focusNonce !== null) setExpanded(true)
  }

  useEffect(() => {
    if (focusNonce !== null) card.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [focusNonce])

  const location = sourceLocation(source)
  const classes = ['evidence-card']
  if (isCited) classes.push('is-cited')
  if (focusNonce !== null) classes.push('is-focused')

  return (
    <li ref={card} className={classes.join(' ')}>
      <button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
        <span className="evidence-meta">
          <span className="evidence-number">{number}</span>
          <span className="evidence-source" title={source.source}>
            {source.source}
          </span>
          {location && <span className="mono">{location}</span>}
          <span className="mono evidence-score" title="Cross-encoder relevance score. Higher is more relevant.">
            {source.rerank_score.toFixed(2)}
          </span>
        </span>
        <span className={expanded ? 'evidence-text' : 'evidence-text is-clamped'}>
          {source.content}
        </span>
        {source.extraction && <span className="evidence-extraction">{source.extraction}</span>}
      </button>
    </li>
  )
}
