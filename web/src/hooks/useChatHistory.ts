import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../service/types'

const MESSAGES_KEY = 'chat_messages'
const PERSIST_DEBOUNCE_MS = 500

export function useChatHistory() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(MESSAGES_KEY)
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  const persistTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (persistTimer.current) clearTimeout(persistTimer.current)
    persistTimer.current = setTimeout(() => {
      localStorage.setItem(MESSAGES_KEY, JSON.stringify(messages))
    }, PERSIST_DEBOUNCE_MS)
    return () => {
      if (persistTimer.current) clearTimeout(persistTimer.current)
    }
  }, [messages])

  const appendUser = useCallback((content: string) => {
    setMessages((prev) => [...prev, { role: 'user', content }])
  }, [])

  const appendAssistantChunk = useCallback((delta: string) => {
    setMessages((prev) => {
      if (prev.length === 0 || prev[prev.length - 1].role !== 'assistant') {
        return [...prev, { role: 'assistant', content: delta }]
      }
      const last = prev[prev.length - 1]
      return [...prev.slice(0, -1), { ...last, content: last.content + delta }]
    })
  }, [])

  const clear = useCallback(() => {
    setMessages([])
    localStorage.removeItem(MESSAGES_KEY)
  }, [])

  return { messages, appendUser, appendAssistantChunk, clear }
}
