import { useCallback, useState } from 'react'

const SESSION_KEY = 'chat_session_id'

function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export function useSession() {
  const [sessionId, setSessionId] = useState<string>(() => {
    const existing = localStorage.getItem(SESSION_KEY)
    if (existing) return existing
    const id = generateId()
    localStorage.setItem(SESSION_KEY, id)
    return id
  })

  const newSession = useCallback(() => {
    const id = generateId()
    localStorage.setItem(SESSION_KEY, id)
    setSessionId(id)
  }, [])

  return { sessionId, newSession }
}
