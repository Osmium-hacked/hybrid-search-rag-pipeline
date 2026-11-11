import { useState } from 'react'
import type { Status, Workspace } from '../api'
import { shortModel } from '../format'
import { GitHubIcon, MoonIcon, StackIcon, SunIcon } from './Icons'

const REPOSITORY = 'https://github.com/Osmium-hacked/hybrid-search-rag-pipeline'

function currentTheme() {
  const chosen = document.documentElement.dataset.theme
  // The dark theme is the default look; light is available from the toggle.
  return chosen === 'light' ? 'light' : 'dark'
}

interface HeaderProps {
  status: Status | null
  workspace: Workspace | null
  onOpenDocuments: () => void
}

export function Header({ status, workspace, onOpenDocuments }: HeaderProps) {
  const [theme, setTheme] = useState(currentTheme)

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    setTheme(next)
    try {
      localStorage.setItem('hyrag-theme', next)
    } catch {
      // The choice just won't persist, e.g. in a private window.
    }
  }

  const documentCount = workspace?.documents.length ?? 0

  return (
    <header className="header">
      <div className="header-inner">
        <a className="wordmark" href="/">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <defs>
              <linearGradient id="wordmark-fill" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="var(--accent-from)" />
                <stop offset="1" stopColor="var(--accent-to)" />
              </linearGradient>
            </defs>
            <rect width="32" height="32" rx="8" fill="url(#wordmark-fill)" />
            <path d="M10 8v16M22 8v16M10 16h12" />
          </svg>
          HyRAG
        </a>
        {status && (
          <dl className="model-pair">
            <div title={status.models.generator}>
              <dt>Writer</dt>
              <dd>{shortModel(status.models.generator)}</dd>
            </div>
            <div title={status.models.judge}>
              <dt>Judge</dt>
              <dd>{shortModel(status.models.judge)}</dd>
            </div>
          </dl>
        )}
        <div className="header-actions">
          <a className="button-icon" href={REPOSITORY} target="_blank" rel="noreferrer noopener" aria-label="Source code on GitHub" title="Source code on GitHub">
            <GitHubIcon />
          </a>
          <button type="button" className="button-quiet" onClick={onOpenDocuments}>
            <StackIcon />
            {workspace ? `${documentCount} ${documentCount === 1 ? 'file' : 'files'}` : 'Files'}
          </button>
          <button
            type="button"
            className="button-icon"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </div>
    </header>
  )
}
