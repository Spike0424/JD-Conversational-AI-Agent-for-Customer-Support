// Backend schemas mirrored for the frontend (keep in sync with app/schemas.py + app/context_models.py)

export interface ShopInfo {
  id: number
  shop_name: string
  shop_logo: string | null
  description: string | null
}

export interface ProductInfo {
  goods_id: number
  goods_name: string
  price: string | null
  thumb_url: string | null
}

export type ContextType =
  | 'text' | 'image' | 'video' | 'goods_card' | 'withdraw'
  | 'auth' | 'system_biz' | 'system_status' | 'mall_system_msg'

export interface ChatContext {
  type: ContextType
  content?: string
  kwargs: {
    shop_id?: string | number
    shop_name?: string
    goods_id?: number
    goods_name?: string
    order_sn?: string
    user_id?: string
    from_uid?: string
    recipient_uid?: string
    media_url?: string
    media_type?: string
    channel_type?: string
  }
}

export interface ProductCard {
  session_id: string
  shop_id: number
  goods_id: number
  goods_name: string
  price: string | null
  price_min: number | null
  price_max: number | null
  thumb_url: string | null
  specifications: Record<string, string>
  message: string
}

export interface ChatResponse {
  session_id: string
  answer: string
  trace_id: string | null
  intent: string
  context_type: string
  citations: unknown[]
  actions: string[]
  need_handoff: boolean
  product_cards: ProductCard[] | null
  metadata: { scene: string; scene_label: string } | null
}

export type ChatMessage =
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string; product_cards?: ProductCard[] }

// ── Sessions (sidebar history) ──────────────────────────────────────

export interface SessionInfo {
  session_id: string
  title: string
  created_at: string
  last_message_at: string
  message_count: number
}

export interface SessionMessagesResponse {
  session_id: string
  messages: { role: string; content: string; timestamp: string }[]
}

// ── Auth ────────────────────────────────────────────────────────────

export interface AuthRequest {
  email: string
  password: string
}

export interface TokenResponse {
  access_token: string
  token_type: 'bearer'
  expires_in: number  // seconds
}
