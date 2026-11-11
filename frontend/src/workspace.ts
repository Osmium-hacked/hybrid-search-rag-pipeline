const STORAGE_KEY = 'hyrag-workspace'

let current: string | null = null

function newId() {
  const bytes = crypto.getRandomValues(new Uint8Array(18))
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_')
}

/**
 * The random ID of this browser's private workspace. Whoever holds it can read
 * the workspace's documents, so it stays in this browser and is never shown.
 */
export function workspaceId() {
  if (current) return current
  try {
    current = localStorage.getItem(STORAGE_KEY)
  } catch {
    // Storage is blocked, e.g. in some private windows.
  }
  if (!current || !/^[A-Za-z0-9_-]{16,64}$/.test(current)) {
    current = newId()
    try {
      localStorage.setItem(STORAGE_KEY, current)
    } catch {
      // The workspace then lasts only until the page is reloaded.
    }
  }
  return current
}
