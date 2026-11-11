import { useState, type ReactNode } from 'react'

interface DropzoneProps {
  onFiles: (files: File[]) => void
  disabled?: boolean
  large?: boolean
  children: ReactNode
}

/**
 * A file picker that also highlights when files are dragged over it. The drop
 * itself is handled by the page-wide listener in App, so a file dropped here
 * isn't added twice.
 */
export function Dropzone({ onFiles, disabled, large, children }: DropzoneProps) {
  const [over, setOver] = useState(false)

  const classes = ['dropzone']
  if (large) classes.push('dropzone-large')
  if (over) classes.push('is-over')
  if (disabled) classes.push('is-disabled')

  return (
    <label
      className={classes.join(' ')}
      onDragEnter={() => !disabled && setOver(true)}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOver(false)
      }}
      onDrop={() => setOver(false)}
    >
      <input
        type="file"
        multiple
        disabled={disabled}
        onChange={(event) => {
          onFiles(Array.from(event.target.files ?? []))
          event.target.value = ''
        }}
      />
      {children}
    </label>
  )
}
