import { useEffect, useState } from 'react'
import { Button, Form, Input, message as antdMessage, Select } from 'antd'
import { fetchProducts, fetchShops } from '../service/api'
import type { ChatContext, ProductInfo, ShopInfo } from '../service/types'

export interface FormState {
  shopId: number
  goodsId: number
  goodsName: string
  orderSn: string
}

interface Props {
  onStart: (state: FormState) => void
}

/**
 * Pre-chat context picker. Embedded inside ChatLayout — the surrounding
 * "Card + title + helper text" lives in the layout, so this just renders
 * the bare form.
 */
export function ConsultationForm({ onStart }: Props) {
  const [shops, setShops] = useState<ShopInfo[]>([])
  const [products, setProducts] = useState<ProductInfo[]>([])
  const [shopId, setShopId] = useState<number | null>(null)
  const [productQuery, setProductQuery] = useState('')
  const [selectedGoodsId, setSelectedGoodsId] = useState<number | null>(null)
  const [selectedGoodsName, setSelectedGoodsName] = useState('')
  const [orderSn, setOrderSn] = useState('')

  useEffect(() => {
    fetchShops()
      .then(setShops)
      .catch(() => antdMessage.error('店铺列表加载失败'))
  }, [])

  useEffect(() => {
    if (shopId === null) return
    if (!productQuery.trim()) {
      fetchProducts(shopId, '').then(setProducts).catch(() => antdMessage.error('商品搜索失败'))
      return
    }
    const timer = setTimeout(() => {
      fetchProducts(shopId, productQuery).then(setProducts).catch(() => antdMessage.error('商品搜索失败'))
    }, 300)
    return () => clearTimeout(timer)
  }, [shopId, productQuery])

  const handleSubmit = () => {
    if (shopId === null) {
      antdMessage.warning('请选择店铺')
      return
    }
    if (selectedGoodsId === null) {
      antdMessage.warning('请选择商品')
      return
    }
    onStart({
      shopId,
      goodsId: selectedGoodsId,
      goodsName: selectedGoodsName,
      orderSn: orderSn.trim(),
    })
  }

  return (
    <Form layout="vertical" onFinish={handleSubmit} style={{ marginTop: 16 }}>
      <Form.Item label="店铺" required style={{ marginBottom: 16 }}>
        <Select
          placeholder="选择店铺"
          value={shopId}
          onChange={(v) => setShopId(v)}
          options={shops.map((s) => ({ value: s.id, label: s.shop_name }))}
        />
      </Form.Item>

      <Form.Item label="商品" required style={{ marginBottom: 16 }}>
        <Input.Search
          placeholder="搜索商品名称"
          value={productQuery}
          onChange={(e) => setProductQuery(e.target.value)}
          enterButton
          style={{ marginBottom: 8 }}
        />
        <Select
          placeholder="选择咨询的商品"
          value={selectedGoodsId}
          onChange={(v, option) => {
            setSelectedGoodsId(v)
            const opt = Array.isArray(option) ? option[0] : option
            setSelectedGoodsName(opt?.label as string || '')
          }}
          options={products.map((p) => ({
            value: p.goods_id,
            label: `${p.goods_name}${p.price ? `（${p.price}）` : ''}`,
          }))}
          showSearch
          optionFilterProp="label"
        />
      </Form.Item>

      <Form.Item label="订单号（选填，售后问题请填写）" style={{ marginBottom: 16 }}>
        <Input
          placeholder="请输入京东订单号"
          value={orderSn}
          onChange={(e) => setOrderSn(e.target.value)}
        />
      </Form.Item>

      <Form.Item style={{ marginTop: 16, marginBottom: 0 }}>
        <Button type="primary" htmlType="submit" block>
          开始咨询
        </Button>
      </Form.Item>
    </Form>
  )
}

export function formStateToContext(state: FormState): ChatContext {
  return {
    type: 'text',
    kwargs: {
      shop_id: String(state.shopId),
      goods_id: state.goodsId,
      goods_name: state.goodsName,
      order_sn: state.orderSn || undefined,
    },
  }
}
