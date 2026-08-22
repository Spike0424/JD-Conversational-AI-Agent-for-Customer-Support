import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import {
  Avatar,
  Button,
  Layout,
  Modal,
  Popover,
  Spin,
  Typography,
  message as antdMessage,
} from 'antd'
import { PlusOutlined, SearchOutlined, SettingOutlined } from '@ant-design/icons'
import { streamChat } from '../service/api'
import { useAuth } from '../hooks/useAuth'
import { useSession } from '../hooks/useSession'
import { AuthForm } from './AuthForm'
import { ChatHeader } from './ChatHeader'
import { ChatInput } from './ChatInput'
import { ConsultationForm, type FormState } from './ConsultationForm'
import { MessageBubble } from './MessageBubble'
import { SessionList } from './SessionList'

const { Sider, Content, Header } = Layout
const { Text, Title } = Typography

type AuthMode = 'login' | 'register'

/**
 * Two-column chat shell (DeepSeek-style). The Sider holds history + new
 * conversation; the Content holds the active chat. A top toolbar hosts the
 * login button / user avatar. AuthForm lives in a Modal that opens on
 * explicit login, on 401 from any authed call, or after ConsultationForm
 * submit when the user hasn't logged in.
 */
export function ChatLayout() {
  const { isLoggedIn, email, login, register, logout } = useAuth()
  const {
    sessions,
    active,
    messages,
    loading,
    refreshSessions,
    startNewSession,
    selectSession,
    clearCurrentConversation,
    removeSession,
    appendUserWithAssistantPlaceholder,
    appendAssistantChunk,
  } = useSession()
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showForm, setShowForm] = useState(!active && isLoggedIn)
  const [showAuth, setShowAuth] = useState(false)
  const [authMode, setAuthMode] = useState<AuthMode>('login')

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, showForm])

  useEffect(() => {
    setShowForm(!active && isLoggedIn)
  }, [active, isLoggedIn])

  const openAuth = (mode: AuthMode = 'login') => {
    setAuthMode(mode)
    setShowAuth(true)
  }

  const handleNewConversation = () => {
    if (streaming) abortRef.current?.abort()
    if (!isLoggedIn) {
      openAuth('login')
      return
    }
    setShowForm(true)
  }

  const handleFormStart = (state: FormState) => {
    startNewSession(state)
    setShowForm(false)
  }

  const handleAuthSubmit = async (email: string, password: string) => {
    if (authMode === 'login') {
      await login(email, password)
    } else {
      await register(email, password)
    }
    setShowAuth(false)
    // Refresh sessions now that we have a token.
    refreshSessions()
  }

  const handleSend = async (text: string) => {
    if (!active) return
    if (!isLoggedIn) {
      // 表单已 gate 登录；理论不会进这里。兜底：弹登录。
      openAuth('login')
      return
    }
    // Atomically append the user message AND an empty assistant placeholder.
    // The placeholder closes the React 18 race where the first SSE chunk
    // arrives before appendUser's setState flushes — without it, the
    // chunk merges into the previous (historical) assistant and the bot
    // reply renders ABOVE the user's question.
    flushSync(() => {
      appendUserWithAssistantPlaceholder(text)
    })
    setStreaming(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      await streamChat({
        sessionId: active.sessionId,
        question: text,
        context: active.context,
        onChunk: appendAssistantChunk,
        signal: ctrl.signal,
      })
      refreshSessions()
    } catch (e: unknown) {
      const err = e as { name?: string; message?: string }
      if (err.name !== 'AbortError') {
        appendAssistantChunk(`\n\n[出错：${err.message}]`)
        antdMessage.error('请求失败：' + err.message)
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  const handleStop = () => abortRef.current?.abort()

  const renderContent = () => {
    if (!isLoggedIn) {
      return (
        <div
          style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 16,
            color: 'var(--text-primary)',
          }}
        >
          <Title level={3} style={{ margin: 0, color: 'var(--text-primary)' }}>
            京东 AI 客服
          </Title>
          <Text style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            登录后即可与 AI 客服对话，了解商品参数、物流、售后等信息。
          </Text>
          <Button
            type="primary"
            size="large"
            onClick={() => openAuth('login')}
            style={{ borderRadius: 16, marginTop: 8 }}
          >
            登录 / 注册
          </Button>
        </div>
      )
    }
    if (showForm) {
      return (
        <div style={{ margin: '64px auto' }}>
          <Title level={4} style={{ marginTop: 0, marginBottom: 4, color: 'var(--text-primary)' }}>
            新建对话
          </Title>
          <Text style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            选择商品后 AI 会基于商品信息回答。
          </Text>
          <div style={{ marginTop: 24 }}>
            <ConsultationForm onStart={handleFormStart} />
          </div>
        </div>
      )
    }
    const lastMsg = messages[messages.length - 1]
    // Show the "thinking" placeholder while streaming AND the assistant
    // bubble is missing OR still empty (the placeholder we created at
    // send-time). As soon as the first SSE chunk arrives, content is
    // non-empty and the placeholder is replaced by the real text.
    const lastIsEmptyAssistant = lastMsg?.role === 'assistant' && !lastMsg.content
    const showThinking = streaming && (!lastMsg || lastIsEmptyAssistant)

    return (
      <>
        {messages.length === 0 && !streaming && (
          <Text style={{ color: 'var(--text-secondary)' }}>
            开始提问吧，例如：{active?.title} 有什么优势？
          </Text>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} msg={m} />
        ))}
        {showThinking && (
          <div style={{
            display: 'flex',
            justifyContent: 'flex-start',
            marginTop: 4,
          }}>
            <div style={{
              padding: '10px 14px',
              borderRadius: 16,
              background: 'var(--bg-elevated)',
              color: 'var(--text-secondary)',
              fontSize: 13,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <Spin size="small" />
              <span>正在思考…</span>
            </div>
          </div>
        )}
      </>
    )
  }

  const handleSelectSession = async (id: string) => {
    if (streaming) abortRef.current?.abort()
    try {
      await selectSession(id)
    } catch {
      antdMessage.error('加载历史失败')
    }
  }

  const userInitial = (email || '?').charAt(0).toUpperCase()

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider
        width={260}
        style={{
          background: 'var(--bg-sider)',
          borderRight: '1px solid var(--bg-elevated)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div style={{ padding: '12px 8px' }}>
          <Button
            type="default"
            icon={<PlusOutlined />}
            block
            onClick={handleNewConversation}
            style={{ borderRadius: 16, height: 36, background: 'var(--bg-elevated)', borderColor: 'var(--bg-elevated)' }}
          >
            开启新对话
          </Button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          <SessionList
            sessions={sessions}
            currentSessionId={active?.sessionId ?? null}
            loading={loading}
            isLoggedIn={isLoggedIn}
            onSelect={handleSelectSession}
            onDelete={(id) => {
              removeSession(id).catch(() => antdMessage.error('删除失败'))
            }}
          />
        </div>

        <div style={{ padding: 8, borderTop: '1px solid var(--bg-elevated)' }}>
          {isLoggedIn ? (
            <Popover
              trigger="hover"
              placement="topRight"
              content={
                <div style={{ padding: 4, minWidth: 200 }}>
                  <Text style={{ display: 'block', marginBottom: 8 }} ellipsis>
                    {email}
                  </Text>
                  <Button type="link" danger onClick={logout} style={{ padding: 0 }}>
                    退出登录
                  </Button>
                </div>
              }
            >
              <Button
                type="text"
                block
                style={{
                  textAlign: 'left',
                  height: 32,
                  padding: '0 8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <Avatar size={20} style={{ background: 'var(--color-primary)', flexShrink: 0 }}>
                  {userInitial}
                </Avatar>
                <Text style={{ fontSize: 13, color: 'var(--text-primary)' }} ellipsis>
                  {email}
                </Text>
              </Button>
            </Popover>
          ) : (
            <Button type="text" block onClick={() => openAuth('login')}
              style={{ height: 32, color: 'var(--text-secondary)', textAlign: 'left' }}>
              <Text style={{ fontSize: 13, color: 'var(--text-secondary)' }}>登录 / 注册</Text>
            </Button>
          )}
        </div>
      </Sider>

      <Layout style={{ background: 'var(--bg-canvas)' }}>
        <Header
          style={{
            background: 'var(--bg-canvas)',
            borderBottom: '1px solid var(--bg-elevated)',
            padding: '0 16px',
            height: 48,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            lineHeight: '48px',
          }}
        >
          <Text style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 600 }}>
            京东客服
          </Text>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {!isLoggedIn && (
              <Button
                type="primary"
                size="small"
                onClick={() => openAuth('login')}
                style={{ borderRadius: 16 }}
              >
                登录
              </Button>
            )}
            <Button
              type="text"
              size="small"
              icon={<SearchOutlined />}
              style={{ color: 'var(--text-secondary)' }}
            />
            <Button
              type="text"
              size="small"
              icon={<SettingOutlined />}
              style={{ color: 'var(--text-secondary)' }}
            />
          </div>
        </Header>

        <Content style={{ display: 'flex', flexDirection: 'column', background: 'var(--bg-canvas)' }}>
          {showForm ? (
            <div style={{ padding: '12px 16px', fontSize: 13 }}>
              <Text style={{ color: 'var(--text-secondary)' }}>新建对话</Text>
            </div>
          ) : (
            <ChatHeader
              title={active?.title ?? ''}
              onClear={clearCurrentConversation}
            />
          )}

          <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
            <div style={{ maxWidth: 800, margin: '0 auto' }}>
            {renderContent()}
          </div>
          </div>

          {!showForm && (
            <ChatInput
              streaming={streaming}
              onSend={handleSend}
              onStop={handleStop}
            />
          )}
        </Content>
      </Layout>

      <Modal
        open={showAuth}
        onCancel={() => setShowAuth(false)}
        footer={null}
        width={400}
        destroyOnClose
        title={null}
      >
        <AuthForm
          mode={authMode}
          onSubmit={handleAuthSubmit}
          onSwitchMode={() =>
            setAuthMode(authMode === 'login' ? 'register' : 'login')
          }
        />
      </Modal>
    </Layout>
  )
}
