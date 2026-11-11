import type { Models, TraceStep } from '../api'
import { formatDuration } from '../format'

const LABELS: Record<TraceStep['agent'], string> = {
  retriever: 'Retrieve',
  generator: 'Write',
  judge: 'Judge',
}

export function Trace({ steps, models }: { steps: TraceStep[]; models: Models }) {
  const retried = steps.some((step) => step.attempt > 1)

  const detail = (step: TraceStep) => {
    if (step.agent === 'generator') return `${step.detail} with ${models.generator}`
    return step.detail
  }

  return (
    <ol className="trace-steps">
      {steps.map((step, index) => (
        <li key={index} className={step.status === 'failed' ? 'trace-failed' : undefined}>
          <span className="trace-label">
            {LABELS[step.agent]}
            {retried && step.agent !== 'retriever' && (
              <span className="mono"> #{step.attempt}</span>
            )}
          </span>
          <span className="trace-detail">{detail(step)}</span>
          <span className="mono">{formatDuration(step.duration_ms)}</span>
        </li>
      ))}
    </ol>
  )
}
