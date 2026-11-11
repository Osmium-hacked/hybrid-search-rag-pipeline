import type { ReactNode } from 'react'

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      className="icon"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

export const CheckIcon = () => (
  <Icon>
    <circle cx="10" cy="10" r="7.5" />
    <path d="m6.8 10.2 2.2 2.2 4.2-4.6" />
  </Icon>
)

export const WarningIcon = () => (
  <Icon>
    <path d="M10 3 2.5 16.5h15L10 3Z" />
    <path d="M10 8.5v3.5M10 14.4v.1" />
  </Icon>
)

export const DashIcon = () => (
  <Icon>
    <circle cx="10" cy="10" r="7.5" />
    <path d="M6.8 10h6.4" />
  </Icon>
)

export const ArrowUpIcon = () => (
  <Icon>
    <path d="M10 16V4M5 9l5-5 5 5" />
  </Icon>
)

export const SunIcon = () => (
  <Icon>
    <circle cx="10" cy="10" r="3.4" />
    <path d="M10 2.5v1.8M10 15.7v1.8M2.5 10h1.8M15.7 10h1.8M4.7 4.7 6 6M14 14l1.3 1.3M4.7 15.3 6 14M14 6l1.3-1.3" />
  </Icon>
)

export const MoonIcon = () => (
  <Icon>
    <path d="M16.5 11.5A7 7 0 0 1 8.5 3.5a7 7 0 1 0 8 8Z" />
  </Icon>
)

export const StackIcon = () => (
  <Icon>
    <path d="m10 3 7 3.5-7 3.5-7-3.5L10 3Z" />
    <path d="m3 10 7 3.5 7-3.5M3 13.5 10 17l7-3.5" />
  </Icon>
)

export const CloseIcon = () => (
  <Icon>
    <path d="m5 5 10 10M15 5 5 15" />
  </Icon>
)

export const UploadIcon = () => (
  <Icon>
    <path d="M10 13V4M6 7.5 10 4l4 3.5M3.5 13v3h13v-3" />
  </Icon>
)

export const PaperclipIcon = () => (
  <Icon>
    <path d="m15.5 9.5-5.8 5.8a3.5 3.5 0 0 1-5-5l6.4-6.4a2.3 2.3 0 0 1 3.3 3.3l-6.2 6.2a1.1 1.1 0 0 1-1.6-1.6l5.6-5.6" />
  </Icon>
)

export const FileIcon = () => (
  <Icon>
    <path d="M11.5 2.5H5.5v15h9V5.5l-3-3Z" />
    <path d="M11.5 2.5v3h3M7.8 10h4.4M7.8 13h4.4" />
  </Icon>
)

export const TrashIcon = () => (
  <Icon>
    <path d="M4 5.5h12M8 5.5V4h4v1.5M5.5 5.5l.8 11h7.4l.8-11" />
  </Icon>
)

export const Spinner = () => (
  <svg className="icon spinner" viewBox="0 0 20 20" fill="none" aria-hidden="true">
    <circle cx="10" cy="10" r="7" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
    <path d="M17 10a7 7 0 0 0-7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
  </svg>
)

export const ImageIcon = () => (
  <Icon>
    <rect x="3" y="4" width="14" height="12" rx="2" />
    <circle cx="7.5" cy="8.5" r="1.3" />
    <path d="m4 15 4.5-4.5 3 3 2-2L17 15" />
  </Icon>
)

export const WaveIcon = () => (
  <Icon>
    <path d="M3 10h1.5M6 7v6M9 4v12M12 7v6M15 8.5v3M17 10h0" />
  </Icon>
)

export const SearchIcon = () => (
  <Icon>
    <circle cx="9" cy="9" r="5.2" />
    <path d="m13 13 4 4" />
  </Icon>
)

export const LayersIcon = () => (
  <Icon>
    <path d="M4 6.5h12M4 10h12M4 13.5h8" />
  </Icon>
)

export const ShieldIcon = () => (
  <Icon>
    <path d="M10 2.8 4 5v4.6c0 3.6 2.5 6.4 6 7.6 3.5-1.2 6-4 6-7.6V5l-6-2.2Z" />
    <path d="m7.4 10 1.8 1.8 3.5-3.6" />
  </Icon>
)

export const PenIcon = () => (
  <Icon>
    <path d="m12.5 4.5 3 3L7 16H4v-3l8.5-8.5Z" />
  </Icon>
)

export const MergeIcon = () => (
  <Icon>
    <path d="M4 4.5c4 0 5 5.5 9 5.5M4 15.5c4 0 5-5.5 9-5.5M13 10h3M14 7.5l2.5 2.5-2.5 2.5" />
  </Icon>
)

export const GitHubIcon = () => (
  <svg className="icon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
  </svg>
)
