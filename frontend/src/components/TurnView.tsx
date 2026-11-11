import { useMemo, useState, type Ref } from 'react'
import Markdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'
import type { AskResult } from '../api'
import type { Turn } from '../App'
import { citedChunks, formatDuration, linkCitations, normalizeMath } from '../format'
import { EvidencePanel } from './EvidencePanel'
import { Trace } from './Trace'
import { VerdictCard } from './VerdictCard'

interface TurnViewProps {
  turn: Turn
  onRetry: () => void
  ref?: Ref<HTMLElement>
}

export function TurnView({ turn, onRetry, ref }: TurnViewProps) {
  return (
    <article className="turn" ref={ref}>
      <h2 className="turn-question">{turn.query}</h2>
      {turn.result ? (
        <TurnResult result={turn.result} />
      ) : turn.error ? (
        <div className="turn-main">
          <div className="error-card" role="alert">
            <p>{turn.error}</p>
            <button type="button" className="button-quiet" onClick={onRetry}>
              Try again
            </button>
          </div>
        </div>
      ) : (
        <div className="turn-main" role="status">
          <div className="working">
            <span className="working-bar" />
            Searching, writing and judging…
          </div>
        </div>
      )}
    </article>
  )
}

function TurnResult({ result }: { result: AskResult }) {
  const [focused, setFocused] = useState<{ id: string; nonce: number } | null>(null)

  const sourceIds = useMemo(() => result.retrieved_sources.map((s) => s.chunk_id), [result])
  const answer = useMemo(
    () => linkCitations(normalizeMath(result.answer), sourceIds),
    [result, sourceIds],
  )
  const cited = useMemo(() => citedChunks(result.answer), [result])

  return (
    <>
      <div className="turn-main">
        <VerdictCard result={result} />
        <div className="answer">
          <Markdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
            // Model output can't load remote images, which could otherwise leak data through the URL.
            disallowedElements={['img']}
            components={{
              a({ href, children }) {
                const id = href?.startsWith('#cite-') ? href.slice('#cite-'.length) : null
                if (id) {
                  const known = sourceIds.includes(id)
                  return (
                    <button
                      type="button"
                      className={known ? 'cite' : 'cite cite-unknown'}
                      disabled={!known}
                      aria-label={known ? `Show source ${children}` : `Unknown source ${id}`}
                      title={known ? id : `${id} wasn't among the retrieved passages`}
                      onClick={() => setFocused({ id, nonce: Date.now() })}
                    >
                      {children}
                    </button>
                  )
                }
                return (
                  <a href={href} target="_blank" rel="noreferrer noopener">
                    {children}
                  </a>
                )
              },
            }}
          >
            {answer}
          </Markdown>
        </div>
        <details className="trace">
          <summary>
            How this answer was made
            <span className="mono">{formatDuration(result.duration_ms)}</span>
          </summary>
          <Trace steps={result.agent_trace} models={result.models} />
        </details>
      </div>
      <EvidencePanel sources={result.retrieved_sources} cited={cited} focused={focused} />
    </>
  )
}
