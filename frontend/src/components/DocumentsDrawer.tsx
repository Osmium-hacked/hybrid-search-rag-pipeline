import { useEffect, useRef, useState } from 'react'
import type { Status, Workspace } from '../api'
import { formatMegabytes, passages } from '../format'
import type { UploadItem } from '../uploads'
import { Dropzone } from './Dropzone'
import { CloseIcon, FileIcon, TrashIcon, UploadIcon } from './Icons'
import { UploadList } from './UploadList'

interface DocumentsDrawerProps {
  open: boolean
  status: Status | null
  workspace: Workspace | null
  uploads: UploadItem[]
  onFiles: (files: File[]) => void
  onDismiss: (id: number) => void
  onRemove: (name: string) => Promise<void>
  onClear: () => Promise<void>
  onClose: () => void
}

export function DocumentsDrawer({
  open,
  status,
  workspace,
  uploads,
  onFiles,
  onDismiss,
  onRemove,
  onClear,
  onClose,
}: DocumentsDrawerProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [pending, setPending] = useState<string | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const element = dialog.current
    if (!element) return
    if (open && !element.open) element.showModal()
    if (!open && element.open) element.close()
  }, [open])

  const run = async (key: string, action: () => Promise<void>) => {
    setPending(key)
    setError(null)
    try {
      await action()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setPending(null)
      setConfirmClear(false)
    }
  }

  const enabled = status?.uploads.enabled ?? false
  const documents = workspace?.documents ?? []

  return (
    <dialog
      ref={dialog}
      className="drawer"
      aria-labelledby="docs-title"
      onClose={() => {
        setConfirmClear(false)
        onClose()
      }}
      onClick={(event) => {
        if (event.target === dialog.current) onClose()
      }}
    >
      <div className="drawer-body">
        <header className="drawer-header">
          <h2 id="docs-title">Your files</h2>
          <button type="button" className="button-icon" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
        </header>

        {documents.length > 0 ? (
          <ul className="doc-list">
            {documents.map((document) => (
              <li key={document.name}>
                <FileIcon />
                <span className="doc-text">
                  <span className="doc-name" title={document.name}>
                    {document.name}
                  </span>
                  <span className="mono">
                    {passages(document.chunks)}
                    {document.extraction && ` · ${document.extraction}`}
                  </span>
                </span>
                <button
                  type="button"
                  className="button-icon"
                  aria-label={`Remove ${document.name}`}
                  title="Remove"
                  disabled={!enabled || pending !== null}
                  onClick={() => run(document.name, () => onRemove(document.name))}
                >
                  <TrashIcon />
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="drawer-muted">No files yet. Add some below.</p>
        )}

        {error && (
          <p className="upload-error" role="alert">
            {error}
          </p>
        )}

        <section className="upload">
          <h3>Add files</h3>
          <Dropzone onFiles={onFiles} disabled={!enabled}>
            <UploadIcon />
            <span>{enabled ? 'Drop files or click to choose' : 'Uploads are unavailable right now'}</span>
            {status && (
              <small>
                Documents, text, images, audio and video · up to{' '}
                {formatMegabytes(status.uploads.max_upload_bytes)} each
              </small>
            )}
          </Dropzone>
          <UploadList items={uploads} onDismiss={onDismiss} compact />
        </section>

        <footer className="drawer-footer">
          {status && (
            <p className="privacy-note">
              Private to this browser. Deleted {status.uploads.retention_days} days after your last change.
            </p>
          )}
          {documents.length > 0 &&
            (confirmClear ? (
              <div className="confirm-row">
                <span>Remove all {documents.length} files?</span>
                <button
                  type="button"
                  className="button-danger"
                  disabled={pending !== null}
                  onClick={() => run('all', onClear)}
                >
                  {pending === 'all' ? 'Removing…' : 'Remove all'}
                </button>
                <button type="button" className="button-quiet" onClick={() => setConfirmClear(false)}>
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                className="link-button"
                disabled={!enabled || pending !== null}
                onClick={() => setConfirmClear(true)}
              >
                Start over with an empty workspace
              </button>
            ))}
        </footer>
      </div>
    </dialog>
  )
}
