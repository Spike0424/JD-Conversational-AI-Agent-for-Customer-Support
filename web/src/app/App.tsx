import { ConfigProvider, theme as antdTheme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { ChatLayout } from '../components/ChatLayout'

function App() {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: antdTheme.darkAlgorithm,
        token: {
          fontSize: 14,
          borderRadius: 12,
          colorPrimary: '#4d9aff',
          colorBgBase: '#000000',
          colorBgContainer: '#0d0d0d',
          colorBgElevated: '#1a1a1a',
          colorBgLayout: '#0a0a0a',
          colorBorder: '#1f1f1f',
          colorBorderSecondary: '#1a1a1a',
        },
      }}
    >
      <ChatLayout />
    </ConfigProvider>
  )
}

export default App
