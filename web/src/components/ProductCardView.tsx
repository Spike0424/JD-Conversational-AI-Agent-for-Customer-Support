import { Card, Image, Typography } from 'antd'
import type { ProductCard } from '../service/types'

const { Text } = Typography

export function ProductCardView({ card }: { card: ProductCard }) {
  return (
    <Card
      size="small"
      style={{ maxWidth: 320, marginTop: 8 }}
      cover={card.thumb_url ? <Image src={card.thumb_url} alt={card.goods_name} style={{ maxHeight: 200, objectFit: 'cover' }} /> : null}
    >
      <Card.Meta
        title={card.goods_name}
        description={
          <>
            {card.price && <Text strong>¥{card.price}</Text>}
            {card.message && <div style={{ marginTop: 4, fontSize: 13 }}>{card.message}</div>}
          </>
        }
      />
      {card.specifications && Object.keys(card.specifications).length > 0 && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
          {Object.entries(card.specifications).slice(0, 4).map(([k, v]) => (
            <div key={k}>{k}: {v}</div>
          ))}
        </div>
      )}
    </Card>
  )
}
