import { workspaceId } from './workspace'

export interface Source {
  chunk_id: string
  source: string
  page: number | null
  location?: string | null
  extraction?: string | null
  rerank_score: number
  content: string
}

export interface TraceStep {
  agent: 'retriever' | 'generator' | 'judge'
  attempt: number
  status: 'passed' | 'failed'
  detail: string
  duration_ms: number
}

export type Verdict = 'supported' | 'unsupported' | 'unavailable'

export interface Models {
  generator: string
  judge: string
}

export interface AskResult {
  answer: string
  citations_verified: boolean
  judge: {
    verdict: Verdict
    reasoning: string
    unsupported_claims: string[]
  }
  models: Models
  retrieved_sources: Source[]
  attempts: number
  agent_trace: TraceStep[]
  duration_ms: number
}

export interface Status {
  models: Models & { vision: string; transcription: string }
  uploads: {
    enabled: boolean
    max_upload_bytes: number
    max_file_chunks: number
    max_workspace_chunks: number
    max_documents: number
    retention_days: number
    extensions: string[]
    formats: Record<string, string>
  }
  sample: string
}

export interface DocumentInfo {
  name: string
  chunks: number
  extraction: string | null
}

export interface Workspace {
  documents: DocumentInfo[]
  total_chunks: number
}

export interface AddedDocument extends DocumentInfo {
  warnings: string[]
}

export interface UploadResult extends Workspace {
  added: AddedDocument[]
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set('X-Workspace-Id', workspaceId())
  let response: Response
  try {
    response = await fetch(path, { ...init, headers })
  } catch {
    throw new Error("Couldn't reach the server. Check your connection and try again.")
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    // FastAPI validation errors put an array in `detail`, not a sentence.
    const detail = typeof body?.detail === 'string' ? body.detail : null
    if (!detail && response.status === 504) {
      throw new Error('This took longer than the server allows. Try a smaller file.')
    }
    throw new Error(detail ?? `The server responded with an error (${response.status}).`)
  }
  return body as T
}

export function getStatus() {
  return request<Status>('/api/v1/status')
}

export function getWorkspace() {
  return request<Workspace>('/api/v1/workspace')
}

export function uploadFile(file: File) {
  const form = new FormData()
  form.append('files', file)
  // One request per file: Vercel caps a request body at 4.5 MB.
  return request<UploadResult>('/api/v1/workspace/documents', { method: 'POST', body: form })
}

export function addSample() {
  return request<UploadResult>('/api/v1/workspace/sample', { method: 'POST' })
}

export function removeDocument(name: string) {
  return request<Workspace>(`/api/v1/workspace/documents?name=${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
}

export function clearWorkspace() {
  return request<Workspace>('/api/v1/workspace', { method: 'DELETE' })
}

export function ask(query: string) {
  return request<AskResult>('/api/v1/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
}
