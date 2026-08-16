import { useState } from 'react'
import { Button, ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { AuthForm } from '../components/AuthForm'
import { ChatRoom } from '../components/ChatRoom'
import { ConsultationForm, formStateToContext, type FormState } from '../components/ConsultationForm'
import { useAuth } from '../hooks/useAuth'
import { useSession } from '../hooks/useSession'

type AuthMode = 'login' | 'register'

function App() {
  const { isLoggedIn, login, register, logout } = useAuth()
  const [authMode, setAuthMode] = useState<AuthMode>('login')
  const { sessionId, newSession } = useSession()
  const [formState, setFormState] = useState<FormState | null>(null)

  if (!isLoggedIn) {
    return (
      <ConfigProvider locale={zhCN}>
        <AuthForm
          mode={authMode}
          onSubmit={async (email, password) => {
            await (authMode === 'login' ? login(email, password) : register(email, password))
          }}
          onSwitchMode={() => setAuthMode(authMode === 'login' ? 'register' : 'login')}
        />
      </ConfigProvider>
    )
  }

  return (
    <ConfigProvider locale={zhCN}>
      {!formState ? (
        <ConsultationForm
          onStart={(s) => setFormState(s)}
        />
      ) : (
        <div>
          <div style={{ textAlign: 'right', padding: 8 }}>
            <Button onClick={logout}>退出登录</Button>
          </div>
          <ChatRoom
            sessionId={sessionId}
            context={formStateToContext(formState)}
            goodsName={formState.goodsName}
            onNewConversation={() => {
              newSession()
              setFormState(null)
            }}
          />
        </div>
      )}
    </ConfigProvider>
  )
}

export default App
