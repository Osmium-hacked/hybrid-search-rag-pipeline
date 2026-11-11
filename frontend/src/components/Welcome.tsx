import type { Status } from '../api'
import { formatMegabytes, shortModel } from '../format'
import type { UploadItem } from '../uploads'
import { Dropzone } from './Dropzone'
import { Flow, FlowLine, FlowNode, FlowStack } from './Flow'
import {
  CheckIcon,
  FileIcon,
  ImageIcon,
  LayersIcon,
  MergeIcon,
  PenIcon,
  SearchIcon,
  ShieldIcon,
  UploadIcon,
  WarningIcon,
  WaveIcon,
} from './Icons'
import { UploadList } from './UploadList'

interface WelcomeProps {
  status: Status | null
  uploads: UploadItem[]
  onFiles: (files: File[]) => void
  onDismiss: (id: number) => void
  onSample: () => void
  sampleBusy: boolean
}

export function Welcome({ status, uploads, onFiles, onDismiss, onSample, sampleBusy }: WelcomeProps) {
  const enabled = status?.uploads.enabled ?? false
  const model = (name: keyof Status['models'], fallback: string) =>
    status ? shortModel(status.models[name]) : fallback

  return (
    <div className="landing">
      <section className="hero">
        <span className="hero-spark hero-spark-1" aria-hidden="true" />
        <span className="hero-spark hero-spark-2" aria-hidden="true" />
        <span className="hero-spark hero-spark-3" aria-hidden="true" />
        <span className="hero-streak" aria-hidden="true" />

        <p className="eyebrow">
          <ShieldIcon />
          Hybrid search RAG with an AI judge
        </p>
        <h1>
          <span className="text-metal">Ask your files anything.</span>{' '}
          <span className="text-accent">Every answer gets judged.</span>
        </h1>
        <p className="hero-lede">
          Upload documents, spreadsheets, slides, images or recordings. HyRAG answers with citations, then a
          second, larger model checks every claim against your files and flags anything they don’t support.
        </p>

        <div className="hero-upload">
          <Dropzone onFiles={onFiles} disabled={!enabled} large>
            <span className="dropzone-badge">
              <UploadIcon />
            </span>
            <span>{enabled ? 'Drop files here, or click to choose' : 'Uploads are unavailable right now'}</span>
            {status && (
              <small>
                Up to {formatMegabytes(status.uploads.max_upload_bytes)} per file ·{' '}
                {status.uploads.max_documents} files per workspace
              </small>
            )}
          </Dropzone>
          <UploadList items={uploads} onDismiss={onDismiss} />
          <div className="hero-actions">
            <button type="button" className="button-glow" disabled={!enabled || sampleBusy} onClick={onSample}>
              {sampleBusy ? 'Adding the sample…' : 'Try it on a sample paper'}
            </button>
            <a className="button-quiet" href="#how">
              How it works
            </a>
          </div>
          {status && (
            <p className="privacy-note">
              Your workspace is private to this browser and deleted {status.uploads.retention_days} days after your
              last change. File text is sent to Groq to read, answer and judge.
            </p>
          )}
        </div>
      </section>

      <section className="features" id="how" aria-labelledby="how-title">
        <h2 id="how-title" className="section-title">
          <span className="text-metal">From any file to a</span> <span className="text-accent">checked answer</span>
        </h2>

        <div className="feature-grid">
          <article className="feature">
            <h3>Reads any file</h3>
            <p>
              Office documents, PDFs, spreadsheets and text are parsed directly. Images and scanned pages are read by a
              vision model, and recordings are transcribed with timestamps.
            </p>
            <Flow label="Documents, images and audio are extracted into passages">
              <FlowStack>
                <FlowNode icon={<FileIcon />} label="Documents" />
                <FlowNode icon={<ImageIcon />} label="Images & scans" meta={model('vision', 'vision')} />
                <FlowNode icon={<WaveIcon />} label="Audio & video" meta={model('transcription', 'whisper')} />
              </FlowStack>
              <FlowLine />
              <FlowNode icon={<LayersIcon />} label="Extract & chunk" meta="1000 chars" hub />
              <FlowLine />
              <FlowNode icon={<SearchIcon />} label="Index" meta="BGE + BM25" />
            </Flow>
          </article>

          <article className="feature">
            <h3>High-precision hybrid search</h3>
            <p>
              Vector search catches meaning, BM25 catches exact names and numbers. Reciprocal Rank Fusion merges both
              and a cross-encoder reranks the shortlist.
            </p>
            <Flow label="A query runs vector and keyword search, which are fused and reranked">
              <FlowNode icon={<SearchIcon />} label="Question" />
              <FlowLine />
              <FlowStack>
                <FlowNode icon={<LayersIcon />} label="Vector" meta="bge-small" />
                <FlowNode icon={<FileIcon />} label="Keyword" meta="BM25" />
              </FlowStack>
              <FlowLine />
              <FlowNode icon={<MergeIcon />} label="Fuse & rerank" meta="RRF · MiniLM" hub />
              <FlowLine />
              <FlowNode icon={<CheckIcon />} label="Top 5" />
            </Flow>
          </article>

          <article className="feature feature-wide">
            <div className="feature-copy">
              <h3>An AI judge for every answer</h3>
              <p>
                One model writes the answer and cites each claim. A separate, larger judge returns a structured verdict
                with every unsupported claim. If it fails, the writer gets one chance to fix it, and if the judge is
                unreachable the answer is marked as not checked, never as verified.
              </p>
            </div>
            <Flow label="The writer drafts an answer, the judge verifies it and can send it back once">
              <FlowNode icon={<LayersIcon />} label="Passages" />
              <FlowLine />
              <FlowNode icon={<PenIcon />} label="Writer" meta={model('generator', 'writer')} />
              <FlowLine />
              <FlowNode icon={<ShieldIcon />} label="Judge" meta={model('judge', 'judge')} hub />
              <FlowLine />
              <FlowStack>
                <span className="flow-verdict is-good">
                  <CheckIcon /> No hallucinations
                </span>
                <span className="flow-verdict is-bad">
                  <WarningIcon /> Claims flagged
                </span>
              </FlowStack>
            </Flow>
          </article>
        </div>

        {status && (
          <dl className="formats">
            {Object.entries(status.uploads.formats).map(([group, list]) => (
              <div key={group}>
                <dt>{group}</dt>
                <dd>{list}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>
    </div>
  )
}
