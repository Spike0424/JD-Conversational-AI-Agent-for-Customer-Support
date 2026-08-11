import { useState } from 'react'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { ChatRoom } from './components/ChatRoom'
import { ConsultationForm, formStateToContext, type FormState } from './components/ConsultationForm'
import { useSession } from './hooks/useSession'

function App() {
  const { sessionId, newSession } = useSession()
  const [formState, setFormState] = useState<FormState | null>(null)

  if (!formState) {
    return (
      <ConfigProvider locale={zhCN}>
        <ConsultationForm onStart={(s) => setFormState(s)} />
      </ConfigProvider>
    )
  }

  return (
    <ConfigProvider locale={zhCN}>
      <ChatRoom
        sessionId={sessionId}
        context={formStateToContext(formState)}
        goodsName={formState.goodsName}
        onNewConversation={() => {
          newSession()
          setFormState(null)
        }}
      />
    </ConfigProvider>
  )
}

export default App
