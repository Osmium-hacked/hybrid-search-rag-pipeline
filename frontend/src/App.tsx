import { useCallback, useEffect, useRef, useState } from 'react'
import {
  addSample,
  ask,
  clearWorkspace,
  getStatus,
  getWorkspace,
  removeDocument,
  type AskResult,
  type Status,
  type Workspace,
} from './api'
import { Composer } from './components/Composer'
import { DocumentsDrawer } from './components/DocumentsDrawer'
import { Header } from './components/Header'
import { UploadIcon } from './components/Icons'
import { Ready } from './components/Ready'
import { TurnView } from './components/TurnView'
import { Welcome } from './components/Welcome'
import { useUploads } from './uploads'

export interface Turn {
  id: number
  query: string
  result?: AskResult
  error?: string
}

const DEFAULT_MAX_BYTES = 4 * 1024 * 1024

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [turns, setTurns] = useState<Turn[]>([])
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [sampleBusy, setSampleBusy] = useState(false)
  const [sampleError, setSampleError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const nextId = useRef(1)
  const latestTurn = useRef<HTMLElement>(null)

  const uploads = useUploads(status?.uploads.max_upload_bytes ?? DEFAULT_MAX_BYTES, setWorkspace)

  const load = useCallback(() => {
    Promise.all([getStatus(), getWorkspace()])
      .then(([nextStatus, nextWorkspace]) => {
        setStatus(nextStatus)
        setWorkspace(nextWorkspace)
        setLoadError(false)
      })
      .catch(() => setLoadError(true))
  }, [])

  useEffect(load, [load])

  const canUpload = status?.uploads.enabled ?? false
  const enqueue = uploads.enqueue

  // Files can be dropped anywhere on the page.
  useEffect(() => {
    if (!canUpload) return
    let depth = 0
    const hasFiles = (event: DragEvent) => event.dataTransfer?.types.includes('Files') ?? false
    const onEnter = (event: DragEvent) => {
      if (!hasFiles(event)) return
      depth += 1
      setDragging(true)
    }
    const onLeave = (event: DragEvent) => {
      if (!hasFiles(event)) return
      depth = Math.max(0, depth - 1)
      if (depth === 0) setDragging(false)
    }
    const onOver = (event: DragEvent) => {
      if (hasFiles(event)) event.preventDefault()
    }
    const onDrop = (event: DragEvent) => {
      depth = 0
      setDragging(false)
      if (!hasFiles(event)) return
      event.preventDefault()
      enqueue(Array.from(event.dataTransfer?.files ?? []))
    }
    window.addEventListener('dragenter', onEnter)
    window.addEventListener('dragleave', onLeave)
    window.addEventListener('dragover', onOver)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('dragenter', onEnter)
      window.removeEventListener('dragleave', onLeave)
      window.removeEventListener('dragover', onOver)
      window.removeEventListener('drop', onDrop)
    }
  }, [canUpload, enqueue])

  const busy = turns.some((turn) => !turn.result && !turn.error)

  const run = useCallback((id: number, query: string) => {
    const finish = (patch: Partial<Turn>) =>
      setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)))
    ask(query)
      .then((result) => finish({ result }))
      .catch((error: Error) => finish({ error: error.message }))
  }, [])

  const submit = useCallback(
    (query: string) => {
      const id = nextId.current++
      setTurns((current) => [...current, { id, query }])
      run(id, query)
    },
    [run],
  )

  const retry = useCallback(
    (turn: Turn) => {
      setTurns((current) =>
        current.map((item) => (item.id === turn.id ? { id: turn.id, query: turn.query } : item)),
      )
      run(turn.id, turn.query)
    },
    [run],
  )

  const loadSample = useCallback(() => {
    setSampleBusy(true)
    setSampleError(null)
    addSample()
      .then(setWorkspace)
      .catch((error: Error) => setSampleError(error.message))
      .finally(() => setSampleBusy(false))
  }, [])

  const remove = useCallback(async (name: string) => {
    setWorkspace(await removeDocument(name))
  }, [])

  const clear = useCallback(async () => {
    setWorkspace(await clearWorkspace())
    setTurns([])
  }, [])

  const turnCount = turns.length
  useEffect(() => {
    if (turnCount > 0) latestTurn.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [turnCount])

  const hasDocuments = (workspace?.documents.length ?? 0) > 0

  let content
  if (turns.length > 0) {
    content = turns.map((turn, index) => (
      <TurnView
        key={turn.id}
        ref={index === turns.length - 1 ? latestTurn : undefined}
        turn={turn}
        onRetry={() => retry(turn)}
      />
    ))
  } else if (workspace && hasDocuments) {
    content = (
      <Ready
        workspace={workspace}
        sample={status?.sample ?? ''}
        uploads={uploads.items}
        onDismiss={uploads.dismiss}
        onAsk={submit}
        onManage={() => setDrawerOpen(true)}
        disabled={busy}
      />
    )
  } else if (workspace || loadError) {
    content = (
      <Welcome
        status={status}
        uploads={uploads.items}
        onFiles={enqueue}
        onDismiss={uploads.dismiss}
        onSample={loadSample}
        sampleBusy={sampleBusy}
      />
    )
  }

  return (
    <div className="app">
      <Header status={status} workspace={workspace} onOpenDocuments={() => setDrawerOpen(true)} />
      <main className="page">
        {loadError && (
          <p className="notice" role="alert">
            The API isn't responding, so uploads and answers won't work until it's back.{' '}
            <button type="button" className="link-button" onClick={load}>
              Try again
            </button>
          </p>
        )}
        {sampleError && (
          <p className="notice" role="alert">
            {sampleError}
          </p>
        )}
        {content}
      </main>
      {hasDocuments && (
        <Composer
          busy={busy}
          onSubmit={submit}
          canUpload={canUpload}
          onFiles={enqueue}
          // Earlier screens show the upload list themselves.
          uploads={turns.length > 0 && !drawerOpen ? uploads.items : undefined}
          onDismiss={uploads.dismiss}
        />
      )}
      <DocumentsDrawer
        open={drawerOpen}
        status={status}
        workspace={workspace}
        uploads={uploads.items}
        onFiles={enqueue}
        onDismiss={uploads.dismiss}
        onRemove={remove}
        onClear={clear}
        onClose={() => setDrawerOpen(false)}
      />
      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          <div>
            <UploadIcon />
            Drop to add these files to your workspace
          </div>
        </div>
      )}
    </div>
  )
}
