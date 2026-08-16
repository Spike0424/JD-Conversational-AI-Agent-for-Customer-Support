import type {
  AuthRequest,
  ChatContext,
  ProductInfo,
  ShopInfo,
  TokenResponse,
} from './types'

const API_BASE = ''  // same-origin in production; Vite proxy in dev

// ── Token storage ─────────────────────────────────────────────────

const TOKEN_KEY = 'chat_token'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(t: string): void {
  localStorage.setItem(TOKEN_KEY, t)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

// ── authedFetch: attach Bearer token, clear on 401 ─────────────────

async function authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const token = getToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const r = await fetch(url, { ...init, headers })
  if (r.status === 401) {
    clearToken()
    throw new Error('未登录或 token 过期')
  }
  return r
}

// ── Public reads (no auth) ─────────────────────────────────────────

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

// ── Auth (register / login) ───────────────────────────────────────

async function postAuth(path: string, body: AuthRequest): Promise<TokenResponse> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    const detail = (err as { detail?: string }).detail
    throw new Error(detail || (path.includes('register') ? '注册失败' : '登录失败'))
  }
  return (await r.json()) as TokenResponse
}

export async function register(email: string, password: string): Promise<TokenResponse> {
  const data = await postAuth('/v1/auth/register', { email, password })
  setToken(data.access_token)
  return data
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const data = await postAuth('/v1/auth/login', { email, password })
  setToken(data.access_token)
  return data
}

// ── Stream chat (requires Bearer token) ──────────────────────────

/**
 * Stream a chat turn via SSE (POST /v1/chat/stream).
 * Calls onChunk for each delta token, returns the full answer.
 */
export async function streamChat(opts: {
  sessionId: string
  question: string
  context: ChatContext
  onChunk: (delta: string) => void
  signal?: AbortSignal
}): Promise<{ answer: string }> {
  const { sessionId, question, context, onChunk, signal } = opts

  const r = await authedFetch(`${API_BASE}/v1/chat/stream`, {
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
      let msg
      try {
        msg = JSON.parse(payload)
      } catch {
        continue  // partial JSON in buffer, wait for next chunk
      }
      if (msg.delta) {
        answer += msg.delta
        onChunk(msg.delta)
        // Yield to React so each chunk renders progressively (otherwise
        // state updates batch into one re-render at end of stream)
        await new Promise((resolve) => setTimeout(resolve, 0))
      }
      if (msg.error_code) {
        throw new Error(msg.message || 'stream error')
      }
    }
  }

  return { answer }
}
