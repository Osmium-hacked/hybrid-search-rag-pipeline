import type { UploadItem } from '../uploads'
import { passages } from '../format'
import { CheckIcon, CloseIcon, Spinner, WarningIcon } from './Icons'

interface UploadListProps {
  items: UploadItem[]
  onDismiss: (id: number) => void
  compact?: boolean
}

function describe(item: UploadItem) {
  switch (item.state) {
    case 'queued':
      return 'Waiting'
    case 'reading':
      return 'Reading and indexing…'
    case 'error':
      return item.message ?? "Couldn't add this file."
    case 'done': {
      const parts = [passages(item.added?.chunks ?? 0)]
      if (item.added?.extraction) parts.push(item.added.extraction)
      return parts.join(' · ')
    }
  }
}

export function UploadList({ items, onDismiss, compact }: UploadListProps) {
  if (items.length === 0) return null

  return (
    <ul className={compact ? 'upload-list is-compact' : 'upload-list'} aria-live="polite">
      {items.map((item) => (
        <li key={item.id} className={`upload-item upload-${item.state}`}>
          <span className="upload-state">
            {item.state === 'done' ? (
              <CheckIcon />
            ) : item.state === 'error' ? (
              <WarningIcon />
            ) : (
              <Spinner />
            )}
          </span>
          <span className="upload-body">
            <span className="upload-name" title={item.name}>
              {item.name}
            </span>
            <span className="upload-detail">{describe(item)}</span>
            {item.added?.warnings.map((warning) => (
              <span key={warning} className="upload-warning">
                {warning}
              </span>
            ))}
          </span>
          {(item.state === 'done' || item.state === 'error') && (
            <button
              type="button"
              className="button-icon upload-dismiss"
              aria-label={`Dismiss ${item.name}`}
              onClick={() => onDismiss(item.id)}
            >
              <CloseIcon />
            </button>
          )}
        </li>
      ))}
    </ul>
  )
}
