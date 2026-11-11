import type { Workspace } from '../api'
import { passages } from '../format'
import type { UploadItem } from '../uploads'
import { UploadList } from './UploadList'

// Written for the sample paper, so they're only offered when it's loaded.
const SAMPLE_QUESTIONS = [
  'How is scaled dot-product attention computed?',
  'Why does the Transformer use positional encodings?',
  'What optimizer and learning rate schedule were used for training?',
  'Why is self-attention preferable to recurrent layers?',
]

function suggestionsFor(workspace: Workspace, sample: string) {
  const names = workspace.documents.map((document) => document.name)
  const suggestions = names.includes(sample) ? [...SAMPLE_QUESTIONS] : []
  for (const name of names.filter((n) => n !== sample).slice(0, 2)) {
    suggestions.push(`Summarize ${name} in a few bullet points.`)
    suggestions.push(`What are the key facts and figures in ${name}?`)
  }
  if (suggestions.length < 4) suggestions.push('What dates, deadlines or amounts are mentioned?')
  return suggestions.slice(0, 5)
}

interface ReadyProps {
  workspace: Workspace
  sample: string
  uploads: UploadItem[]
  onDismiss: (id: number) => void
  onAsk: (query: string) => void
  onManage: () => void
  disabled: boolean
}

export function Ready({ workspace, sample, uploads, onDismiss, onAsk, onManage, disabled }: ReadyProps) {
  const count = workspace.documents.length

  return (
    <section className="empty">
      <h1>
        <span className="text-metal">Ready.</span> <span className="text-accent">Ask away.</span>
      </h1>
      <p className="empty-lede">
        {count} {count === 1 ? 'file is' : 'files are'} indexed as {passages(workspace.total_chunks)}. Each answer
        is checked by the judge before you see the verdict.
      </p>

      <ul className="doc-chips">
        {workspace.documents.map((document) => (
          <li key={document.name} title={document.extraction ?? undefined}>
            {document.name}
          </li>
        ))}
        <li>
          <button type="button" className="link-button" onClick={onManage}>
            Add or remove files
          </button>
        </li>
      </ul>

      <UploadList items={uploads} onDismiss={onDismiss} />

      <div className="starters">
        <h2>Try asking</h2>
        <ul>
          {suggestionsFor(workspace, sample).map((question) => (
            <li key={question}>
              <button type="button" disabled={disabled} onClick={() => onAsk(question)}>
                {question}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
