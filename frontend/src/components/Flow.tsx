import type { ReactNode } from 'react'

/**
 * Small pipeline diagrams for the landing page: boxes joined by dashed lines.
 * Built from HTML rather than SVG so labels stay readable at phone width,
 * where the row turns into a column.
 */

interface NodeProps {
  icon: ReactNode
  label: string
  meta?: string
  hub?: boolean
}

export function FlowNode({ icon, label, meta, hub }: NodeProps) {
  return (
    <div className={hub ? 'flow-node is-hub' : 'flow-node'}>
      <span className="flow-icon">{icon}</span>
      <span className="flow-label">{label}</span>
      {meta && <span className="flow-meta">{meta}</span>}
    </div>
  )
}

export function FlowStack({ children }: { children: ReactNode }) {
  return <div className="flow-stack">{children}</div>
}

export function FlowLine() {
  return <span className="flow-line" aria-hidden="true" />
}

export function Flow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flow" role="img" aria-label={label}>
      {children}
    </div>
  )
}
