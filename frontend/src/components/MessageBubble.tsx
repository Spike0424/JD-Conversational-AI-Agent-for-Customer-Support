import { Typography } from 'antd'
import type { ChatMessage } from '../types'
import { ProductCardView } from './ProductCardView'

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12,
    }}>
      <div style={{
        maxWidth: '70%',
        padding: '8px 14px',
        borderRadius: 8,
        background: isUser ? '#1677ff' : '#f0f0f0',
        color: isUser ? '#fff' : '#000',
      }}>
        <Typography.Text style={{ color: 'inherit', whiteSpace: 'pre-wrap' }}>
          {msg.content}
        </Typography.Text>
        {!isUser && 'product_cards' in msg && msg.product_cards?.map((c, i) => (
          <ProductCardView key={i} card={c} />
        ))}
      </div>
    </div>
  )
}
