import { useState } from 'react'
import { Button, Card, Form, Input, Typography } from 'antd'

const { Title } = Typography

type Mode = 'login' | 'register'

interface AuthFormProps {
  mode: Mode
  onSubmit: (email: string, password: string) => Promise<void>
  onSwitchMode: () => void
}

export function AuthForm({ mode, onSubmit, onSwitchMode }: AuthFormProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    try {
      await onSubmit(email, password)
    } catch (e: unknown) {
      const err = e as { message?: string }
      setError(err.message || (mode === 'login' ? '登录失败' : '注册失败'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card style={{ maxWidth: 400, margin: '80px auto' }}>
      <Title level={3} style={{ textAlign: 'center' }}>
        {mode === 'login' ? '登录' : '注册'}
      </Title>
      <Form layout="vertical" onFinish={handleSubmit} disabled={loading}>
        <Form.Item
          label="邮箱"
          validateStatus={error ? 'error' : ''}
          help={error || ''}
        >
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />
        </Form.Item>
        <Form.Item label="密码（8-64 字符）">
          <Input.Password
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            {mode === 'login' ? '登录' : '注册'}
          </Button>
        </Form.Item>
        <Button type="link" onClick={onSwitchMode} block>
          {mode === 'login' ? '没有账号？去注册' : '已有账号？去登录'}
        </Button>
      </Form>
    </Card>
  )
}
