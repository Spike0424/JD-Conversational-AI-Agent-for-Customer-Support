import { useState } from 'react'
import { Button, Popconfirm, Typography } from 'antd'
import { DeleteOutlined } from '@ant-design/icons'
import type { SessionInfo } from '../service/types'

const { Text } = Typography

interface Props {
  sessions: SessionInfo[]
  currentSessionId: string | null
  loading: boolean
  isLoggedIn: boolean
  onSelect: (sessionId: string) => void
  onDelete: (sessionId: string) => void
}

interface Group {
  label: string
  sessions: SessionInfo[]
}

function groupByDate(sessions: SessionInfo[]): Group[] {
  const now = Date.now()
  const day = 24 * 60 * 60 * 1000
  const today: SessionInfo[] = []
  const week: SessionInfo[] = []
  const month: SessionInfo[] = []
  for (const s of sessions) {
    const ts = new Date(s.last_message_at).getTime()
    const age = now - ts
    if (age < day) today.push(s)
    else if (age < 7 * day) week.push(s)
    else month.push(s)
  }
  const out: Group[] = []
  if (today.length) out.push({ label: '今天', sessions: today })
  if (week.length) out.push({ label: '7 天内', sessions: week })
  if (month.length) out.push({ label: '30 天内', sessions: month })
  return out
}

function SessionRow({
  session,
  active,
  deletable,
  onSelect,
  onDelete,
}: {
  session: SessionInfo
  active: boolean
  deletable: boolean
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onSelect(session.session_id)}
      style={{
        cursor: 'pointer',
        height: 32,
        padding: '0 8px',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        borderRadius: 6,
        background: active ? '#1a1a1a' : hovered ? '#1a1a1a' : 'transparent',
      }}
    >
      <Text
        ellipsis
        style={{
          flex: 1,
          fontSize: 13,
          color: active ? '#ffffff' : '#a0a0a0',
          fontWeight: active ? 500 : 400,
        }}
      >
        {session.title}
      </Text>
      {deletable && (
        <Popconfirm
          title="删除该对话？"
          description="删除后无法恢复。"
          okText="删除"
          cancelText="取消"
          onConfirm={(e) => {
            e?.stopPropagation()
            onDelete(session.session_id)
          }}
          onCancel={(e) => e?.stopPropagation()}
        >
          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            onClick={(e) => e.stopPropagation()}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            style={{
              width: 20,
              height: 20,
              padding: 0,
              color: hovered ? '#ffffff' : '#5e5e5e',
              opacity: hovered ? 1 : 0,
              transition: 'opacity 120ms ease',
              pointerEvents: hovered ? 'auto' : 'none',
            }}
          />
        </Popconfirm>
      )}
    </div>
  )
}

/**
 * Sidebar history grouped by date (today / 7d / 30d). Renders demo entries
 * when the user is not logged in so the panel isn't empty.
 */
export function SessionList({
  sessions,
  currentSessionId,
  loading,
  isLoggedIn,
  onSelect,
  onDelete,
}: Props) {
  if (loading) {
    return (
      <Text style={{ padding: 12, fontSize: 13, color: '#a0a0a0' }}>加载中…</Text>
    )
  }
  const groups = groupByDate(sessions)
  if (groups.length === 0) {
    return (
      <Text style={{ padding: 12, fontSize: 13, color: '#a0a0a0' }}>
        暂无历史对话
      </Text>
    )
  }
  return (
    <div style={{ padding: '0 8px' }}>
      {groups.map((g) => (
        <div key={g.label} style={{ marginBottom: 4 }}>
          <div
            style={{
              padding: '8px 8px 4px',
              fontSize: 12,
              color: '#5e5e5e',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span>{g.label}</span>
          </div>
          {g.sessions.map((s) => (
            <SessionRow
              key={s.session_id}
              session={s}
              active={s.session_id === currentSessionId}
              deletable={isLoggedIn && !s.session_id.startsWith('demo-')}
              onSelect={onSelect}
              onDelete={onDelete}
            />
          ))}
        </div>
      ))}
    </div>
  )
}
