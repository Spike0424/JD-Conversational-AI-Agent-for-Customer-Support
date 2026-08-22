import { useState } from 'react'
import { Button, Popconfirm, Tooltip, Typography } from 'antd'
import { DeleteOutlined } from '@ant-design/icons'

const { Text } = Typography

interface Props {
  title: string
  onClear: () => void
}

/**
 * Top bar of the chat area: conversation title on the left, clear button
 * on the right. The clear control is hidden until hover so the bar stays
 * quiet during reading.
 */
export function ChatHeader({ title, onClear }: Props) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        height: 48,
        padding: '0 16px',
        borderBottom: '1px solid var(--bg-elevated)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'transparent',
      }}
    >
      <Text strong style={{ fontSize: 14, color: 'var(--text-primary)' }} ellipsis>
        {title || '(未选择)'}
      </Text>
      {hovered && (
        <Popconfirm
          title="清空当前对话？"
          description="仅清空当前显示的消息，下一次提问仍会保留在同一对话里。"
          okText="清空"
          cancelText="取消"
          onConfirm={onClear}
        >
          <Tooltip title="清空当前对话" placement="bottom">
            <Button
              type="text"
              size="small"
              icon={<DeleteOutlined />}
              style={{ color: 'var(--text-secondary)' }}
            />
          </Tooltip>
        </Popconfirm>
      )}
    </div>
  )
}
