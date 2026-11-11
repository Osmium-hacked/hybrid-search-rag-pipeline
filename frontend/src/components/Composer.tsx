import { useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import type { UploadItem } from '../uploads'
import { ArrowUpIcon, PaperclipIcon } from './Icons'
import { UploadList } from './UploadList'

const MAX_QUERY_LENGTH = 2000

interface ComposerProps {
  busy: boolean
  onSubmit: (query: string) => void
  canUpload: boolean
  onFiles: (files: File[]) => void
  uploads?: UploadItem[]
  onDismiss: (id: number) => void
}

export function Composer({ busy, onSubmit, canUpload, onFiles, uploads = [], onDismiss }: ComposerProps) {
  const [value, setValue] = useState('')
  const textarea = useRef<HTMLTextAreaElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const query = value.trim()
  const canSubmit = query.length > 0 && !busy

  useLayoutEffect(() => {
    const element = textarea.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${element.scrollHeight}px`
  }, [value])

  const submit = (event?: FormEvent) => {
    event?.preventDefault()
    if (!canSubmit) return
    onSubmit(query)
    setValue('')
  }

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer-dock">
      {uploads.length > 0 && (
        <div className="composer-uploads">
          <UploadList items={uploads} onDismiss={onDismiss} compact />
        </div>
      )}
      <form className="composer" onSubmit={submit}>
        <button
          type="button"
          className="button-icon attach"
          aria-label="Add files"
          title="Add files"
          disabled={!canUpload}
          onClick={() => fileInput.current?.click()}
        >
          <PaperclipIcon />
        </button>
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(event) => {
            onFiles(Array.from(event.target.files ?? []))
            event.target.value = ''
          }}
        />
        <textarea
          ref={textarea}
          rows={1}
          value={value}
          maxLength={MAX_QUERY_LENGTH}
          placeholder="Ask a question about your files"
          aria-label="Your question"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <button type="submit" className="send" disabled={!canSubmit} aria-label="Ask">
          <ArrowUpIcon />
        </button>
      </form>
      <p className="composer-hint">
        Each question is answered on its own, from your files only. Drop files anywhere to add them.
      </p>
    </div>
  )
}
