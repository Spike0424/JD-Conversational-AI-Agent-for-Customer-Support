import { useRef, useState } from 'react'
import { Button, Input, message as antdMessage, Tooltip } from 'antd'
import { ArrowUpOutlined, PaperClipOutlined, StopOutlined } from '@ant-design/icons'

interface Props {
  streaming: boolean
  onSend: (text: string) => Promise<void> | void
  onStop: () => void
}

/**
 * Composer at the bottom of the chat area (DeepSeek-style rounded pill).
 * - Enter sends; Shift+Enter inserts a newline.
 * - The send button is a 32px circle; while streaming it becomes a stop button.
 */
export function ChatInput({ streaming, onSend, onStop }: Props) {
  const [value, setValue] = useState('')
  const sendingRef = useRef(false)

  const handleSend = async () => {
    const text = value.trim()
    if (!text || streaming || sendingRef.current) return
    sendingRef.current = true
    setValue('')
    try {
      await onSend(text)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      antdMessage.error('发送失败：' + msg)
    } finally {
      sendingRef.current = false
    }
  }

  return (
    <div style={{ padding: 16, background: 'var(--bg-canvas)' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 8,
          background: 'var(--bg-elevated)',
          borderRadius: 24,
          padding: '10px 12px',
          maxWidth: 800,
          margin: '0 auto',
        }}
      >
        <Tooltip title="附件">
          <Button
            type="text"
            size="small"
            icon={<PaperClipOutlined />}
            style={{ color: 'var(--text-secondary)', flexShrink: 0 }}
          />
        </Tooltip>
        <Input.TextArea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="输入您的问题...（Shift+Enter 换行，Enter 发送）"
          autoSize={{ minRows: 1, maxRows: 6 }}
          disabled={streaming}
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault()
              handleSend()
            }
          }}
          variant="borderless"
          style={{ background: 'transparent', resize: 'none' }}
        />
        {streaming ? (
          <Tooltip title="停止生成">
            <Button
              danger
              onClick={onStop}
              icon={<StopOutlined />}
              style={{ width: 32, height: 32, borderRadius: 16, flexShrink: 0 }}
            />
          </Tooltip>
        ) : (
          <Button
            type="primary"
            onClick={handleSend}
            icon={<ArrowUpOutlined />}
            style={{ width: 32, height: 32, borderRadius: 16, flexShrink: 0 }}
          />
        )}
      </div>
    </div>
  )
}
