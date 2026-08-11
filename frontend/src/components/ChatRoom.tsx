import { useRef, useState } from 'react'
import { Button, Input, Space, Spin, Typography, message as antdMessage } from 'antd'
import { streamChat } from '../api'
import { useChatHistory } from '../hooks/useChatHistory'
import type { ChatContext } from '../types'
import { MessageBubble } from './MessageBubble'

const { Text } = Typography

interface Props {
  sessionId: string
  context: ChatContext
  goodsName: string
  onNewConversation: () => void
}

export function ChatRoom({ sessionId, context, goodsName, onNewConversation }: Props) {
  const { messages, appendUser, appendAssistantChunk, clear } = useChatHistory()
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const send = async () => {
    const question = input.trim()
    if (!question || streaming) return
    setInput('')
    appendUser(question)
    setStreaming(true)

    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      await streamChat({
        sessionId,
        question,
        context,
        history: messages,
        onChunk: (delta) => {
          appendAssistantChunk(delta)
          scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
        },
        signal: ctrl.signal,
      })
    } catch (e: unknown) {
      const err = e as { name?: string; message?: string }
      if (err.name === 'AbortError') {
        antdMessage.info('已停止生成')
      } else {
        appendAssistantChunk(`\n\n[出错：${err.message}]`)
        antdMessage.error('请求失败：' + err.message)
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  const stop = () => {
    abortRef.current?.abort()
  }

  const handleNewConversation = () => {
    if (streaming) abortRef.current?.abort()
    clear()
    onNewConversation()
  }

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #f0f0f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text strong>当前咨询：{goodsName}</Text>
        <Button onClick={handleNewConversation} size="small">新对话</Button>
      </div>

      <div ref={scrollRef} style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {messages.length === 0 && (
          <Text type="secondary">请输入您的问题，例如：{goodsName}有什么优势？</Text>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} msg={m} />
        ))}
        {streaming && <Spin size="small" style={{ marginLeft: 16 }} />}
      </div>

      <div style={{ padding: 16, borderTop: '1px solid #f0f0f0' }}>
        <Space.Compact style={{ width: '100%' }}>
          <Input.TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入您的问题..."
            autoSize={{ minRows: 1, maxRows: 4 }}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            disabled={streaming}
          />
          {streaming ? (
            <Button danger onClick={stop} style={{ width: 100 }}>停止</Button>
          ) : (
            <Button type="primary" onClick={send} style={{ width: 100 }}>发送</Button>
          )}
        </Space.Compact>
      </div>
    </div>
  )
}
