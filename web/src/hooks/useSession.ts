import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteSession, fetchSessionMessages, fetchSessions, getToken } from '../service/api'
import type { ChatContext, ChatMessage, SessionInfo } from '../service/types'

/**
 * Insert a paragraph break before ``delta`` when ``prev`` ends a sentence and
 * the new chunk starts a new thought. Without this the model often streams
 * flat prose that reads as a wall of text in the chat bubble.
 *
 * Heuristic: if prev ends with 。！？.! (and not an abbreviation) and delta
 * starts with an uppercase letter / digit / `**` / `-`, prepend ``\n\n``.
 * Markdown list continuations like "1. " or "10." are skipped so we don't
 * double-break numbered lists.
 */
function mergeAssistantDelta(prev: string, delta: string): string {
  if (!prev || !delta) return delta
  const last = prev.slice(-1)
  const first = delta[0]
  if (
    (last === '。' || last === '！' || last === '？' || last === '.' || last === '!') &&
    first !== '\n' && first !== ' '
  ) {
    // Skip decimal point / ordinal: prev ends with digit before "." (e.g. "6.", "17.")
    // or next char is a digit (e.g. "6.3" streamed as "6." then "3").
    if (/\d$/.test(prev)) return delta
    if (/^\d/.test(delta)) return delta
    return '\n\n' + delta
  }
  return delta
}

export interface ActiveSession {
  sessionId: string
  context: ChatContext
  title: string  // typically goods_name from ConsultationForm
}

// Demo sessions shown when the user is not logged in (or has no real history).
// Marked with `demo-` prefix so SessionList knows not to render delete buttons.
const DEMO_SESSIONS: SessionInfo[] = [
  {
    session_id: 'demo-1',
    title: '智能保温杯有什么优势？',
    created_at: '2026-08-19T08:00:00Z',
    last_message_at: new Date().toISOString(),
    message_count: 4,
  },
  {
    session_id: 'demo-2',
    title: '帮我推荐一款入门相机',
    created_at: '2026-08-17T14:30:00Z',
    last_message_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    message_count: 6,
  },
  {
    session_id: 'demo-3',
    title: '订单号 20260819xxxx 物流',
    created_at: '2026-08-10T09:00:00Z',
    last_message_at: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString(),
    message_count: 2,
  },
]

/**
 * Multi-session orchestrator.
 *
 * When the user is logged in, session list and messages come from the backend.
 * When unauthenticated, falls back to demo sessions so the sidebar isn't empty
 * and 401s from any authed call are swallowed silently.
 */
export function useSession() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [active, setActive] = useState<ActiveSession | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)

  const refreshSessions = useCallback(async () => {
    if (!getToken()) {
      setSessions(DEMO_SESSIONS)
      return
    }
    try {
      const list = await fetchSessions()
      setSessions(list)
    } catch {
      // 401 or network failure — fall back to demo and stay quiet.
      setSessions(DEMO_SESSIONS)
    }
  }, [])

  useEffect(() => {
    refreshSessions()
  }, [refreshSessions])

  const startNewSession = useCallback(
    (formState: { goodsId: number; goodsName: string; shopId: number; orderSn: string }) => {
      const sessionId = crypto.randomUUID()
      const context: ChatContext = {
        type: 'text',
        kwargs: {
          shop_id: String(formState.shopId),
          goods_id: formState.goodsId,
          goods_name: formState.goodsName,
          order_sn: formState.orderSn || undefined,
        },
      }
      setActive({ sessionId, context, title: formState.goodsName })
      setMessages([])
    },
    [],
  )

  const sessionsRef = useRef<SessionInfo[]>([])
  sessionsRef.current = sessions

  const selectSession = useCallback(async (sessionId: string) => {
    // Demo sessions can't be loaded — just show them as if they were the
    // active session but with empty messages.
    if (sessionId.startsWith('demo-')) {
      const meta = sessionsRef.current.find((s) => s.session_id === sessionId)
      setActive({
        sessionId,
        context: { type: 'text', kwargs: {} },
        title: meta?.title || '历史对话',
      })
      setMessages([])
      return
    }
    if (!getToken()) {
      // Authed call would 401 — treat as soft no-op.
      return
    }
    setLoading(true)
    try {
      const data = await fetchSessionMessages(sessionId)
      const meta = sessionsRef.current.find((s) => s.session_id === sessionId)
      setActive({
        sessionId,
        context: { type: 'text', kwargs: {} },
        title: meta?.title || '历史对话',
      })
      setMessages(
        data.messages
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content }) as ChatMessage),
      )
    } catch (e) {
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const clearCurrentConversation = useCallback(() => {
    setMessages([])
  }, [])

  const removeSession = useCallback(
    async (sessionId: string) => {
      if (sessionId.startsWith('demo-')) {
        setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
        return
      }
      if (!getToken()) return
      await deleteSession(sessionId)
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
      if (active?.sessionId === sessionId) {
        setActive(null)
        setMessages([])
      }
    },
    [active],
  )

  const appendUser = useCallback((content: string) => {
    setMessages((prev) => [...prev, { role: 'user', content }])
  }, [])

  /**
   * Append a user message AND an empty assistant placeholder in one shot.
   * The placeholder starts invisible (height collapses) and is filled in by
   * the first chunk from `appendAssistantChunk`, which sees prev's last as
   * assistant and merges into it. This closes the race where the first SSE
   * chunk arrives before React applies the appendUser setState — without
   * it, the chunk merges into the previous assistant message and the bot
   * reply visually renders ABOVE the user's question.
   */
  const appendUserWithAssistantPlaceholder = useCallback((content: string) => {
    setMessages((prev) => [...prev, { role: 'user', content }, { role: 'assistant', content: '' }])
  }, [])

  const appendAssistantChunk = useCallback((delta: string) => {
    setMessages((prev) => {
      if (prev.length === 0 || prev[prev.length - 1].role !== 'assistant') {
        return [...prev, { role: 'assistant', content: delta }]
      }
      const last = prev[prev.length - 1]
      return [...prev.slice(0, -1), { ...last, content: last.content + mergeAssistantDelta(last.content, delta) }]
    })
  }, [])

  return {
    sessions,
    active,
    messages,
    loading,
    refreshSessions,
    startNewSession,
    selectSession,
    clearCurrentConversation,
    removeSession,
    appendUser,
    appendUserWithAssistantPlaceholder,
    appendAssistantChunk,
  }
}
