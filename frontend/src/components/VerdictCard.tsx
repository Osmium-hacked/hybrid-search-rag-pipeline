import type { AskResult, Verdict } from '../api'
import { CheckIcon, DashIcon, WarningIcon } from './Icons'

const COPY: Record<Verdict, { title: string; icon: () => React.JSX.Element }> = {
  supported: { title: 'No hallucinations found', icon: CheckIcon },
  unsupported: { title: 'Possible hallucination', icon: WarningIcon },
  unavailable: { title: 'Not checked for hallucinations', icon: DashIcon },
}

function attemptNote({ judge, attempts }: AskResult) {
  if (judge.verdict === 'supported' && attempts > 1) {
    return `Passed on attempt ${attempts}, after the writer corrected what the judge flagged.`
  }
  if (judge.verdict === 'unsupported' && attempts > 1) {
    return `Still failing after ${attempts} attempts. Read this answer with caution.`
  }
  return null
}

export function VerdictCard({ result }: { result: AskResult }) {
  const { judge, models } = result
  const { title, icon: Icon } = COPY[judge.verdict]
  const note = attemptNote(result)

  return (
    <section className={`verdict verdict-${judge.verdict}`} aria-label="Judge verdict">
      <header>
        <Icon />
        <strong>{title}</strong>
        <span className="mono" title="The judge model, separate from the writer">
          Judge · {models.judge}
        </span>
      </header>
      <p>{judge.reasoning}</p>
      {note && <p className="verdict-note">{note}</p>}
      {judge.unsupported_claims.length > 0 && (
        <p className="verdict-label">
          {judge.unsupported_claims.length === 1
            ? "This claim isn't backed by your files:"
            : `These ${judge.unsupported_claims.length} claims aren't backed by your files:`}
        </p>
      )}
      {judge.unsupported_claims.length > 0 && (
        <ul>
          {judge.unsupported_claims.map((claim) => (
            <li key={claim}>{claim}</li>
          ))}
        </ul>
      )}
    </section>
  )
}
