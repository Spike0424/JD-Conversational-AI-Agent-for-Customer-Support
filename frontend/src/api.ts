import type { ChatContext, ChatMessage, ProductInfo, ShopInfo } from './types'

const API_BASE = ''  // same-origin in production; Vite proxy in dev

export async function fetchShops(): Promise<ShopInfo[]> {
  const r = await fetch(`${API_BASE}/v1/shops`)
  if (!r.ok) throw new Error(`fetchShops failed: ${r.status}`)
  return r.json()
}

export async function fetchProducts(shopId: number, q: string): Promise<ProductInfo[]> {
  const params = new URLSearchParams({ shop_id: String(shopId), q })
  const r = await fetch(`${API_BASE}/v1/products?${params}`)
  if (!r.ok) throw new Error(`fetchProducts failed: ${r.status}`)
  return r.json()
}

/**
 * Stream a chat turn via SSE (POST /v1/chat/stream).
 * Calls onChunk for each delta token, returns the full answer.
 */
export async function streamChat(opts: {
  sessionId: string
  question: string
  context: ChatContext
  history: ChatMessage[]
  onChunk: (delta: string) => void
  signal?: AbortSignal
}): Promise<{ answer: string }> {
  const { sessionId, question, context, onChunk, signal } = opts

  const r = await fetch(`${API_BASE}/v1/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question, context }),
    signal,
  })

  if (!r.ok || !r.body) {
    throw new Error(`streamChat failed: ${r.status}`)
  }

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let answer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE: lines start with "data: ", separated by "\n\n"
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''

    for (const evt of events) {
      const line = evt.trim()
      if (!line.startsWith('data: ')) continue
      const payload = line.slice(6)
      if (payload === '[DONE]') continue
      try {
        const msg = JSON.parse(payload)
        if (msg.delta) {
          answer += msg.delta
          onChunk(msg.delta)
        }
        if (msg.error_code) {
          throw new Error(msg.message || 'stream error')
        }
      } catch (e) {
        // partial JSON in buffer, skip - will be completed in next chunk
      }
    }
  }

  return { answer }
}
