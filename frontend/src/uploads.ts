import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import { uploadFile, type AddedDocument, type Workspace } from './api'
import { formatMegabytes } from './format'

const DONE_VISIBLE_MS = 6000

export interface UploadItem {
  id: number
  name: string
  state: 'queued' | 'reading' | 'done' | 'error'
  message?: string
  added?: AddedDocument
}

/** Uploads files one at a time, in order, and tracks how each one went. */
export function useUploads(maxBytes: number, onWorkspace: (workspace: Workspace) => void) {
  const [items, setItems] = useState<UploadItem[]>([])
  const queue = useRef<{ id: number; file: File }[]>([])
  const running = useRef(false)
  const nextId = useRef(1)
  const latestOnWorkspace = useRef(onWorkspace)
  useLayoutEffect(() => {
    latestOnWorkspace.current = onWorkspace
  })

  const update = (id: number, patch: Partial<UploadItem>) =>
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)))

  const pump = useCallback(async () => {
    if (running.current) return
    running.current = true
    while (queue.current.length > 0) {
      const { id, file } = queue.current.shift()!
      update(id, { state: 'reading' })
      try {
        const result = await uploadFile(file)
        latestOnWorkspace.current(result)
        update(id, { state: 'done', added: result.added[0] })
        // Successes clear themselves; errors and warnings stay until dismissed.
        if (!result.added[0]?.warnings.length) {
          setTimeout(() => setItems((current) => current.filter((item) => item.id !== id)), DONE_VISIBLE_MS)
        }
      } catch (error) {
        update(id, { state: 'error', message: (error as Error).message })
      }
    }
    running.current = false
  }, [])

  const enqueue = useCallback(
    (files: File[]) => {
      const added: UploadItem[] = []
      for (const file of files) {
        const id = nextId.current++
        if (file.size > maxBytes) {
          added.push({
            id,
            name: file.name,
            state: 'error',
            message: `Larger than the ${formatMegabytes(maxBytes)} limit.`,
          })
        } else if (file.size === 0) {
          added.push({ id, name: file.name, state: 'error', message: 'The file is empty.' })
        } else {
          added.push({ id, name: file.name, state: 'queued' })
          queue.current.push({ id, file })
        }
      }
      // Newest first, so the file just dropped is the one in view.
      setItems((current) => [...added.reverse(), ...current])
      void pump()
    },
    [maxBytes, pump],
  )

  const dismiss = useCallback((id: number) => {
    setItems((current) => current.filter((item) => item.id !== id))
  }, [])

  const clearFinished = useCallback(() => {
    setItems((current) => current.filter((item) => item.state === 'queued' || item.state === 'reading'))
  }, [])

  const busy = items.some((item) => item.state === 'queued' || item.state === 'reading')

  return { items, enqueue, dismiss, clearFinished, busy }
}
