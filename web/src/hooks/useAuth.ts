import { useCallback, useState } from 'react'
import {
  clearToken,
  getToken,
  login as apiLogin,
  register as apiRegister,
  setToken,
} from '../service/api'

export function useAuth() {
  const [token, setTokenState] = useState<string | null>(getToken())
  const isLoggedIn = !!token

  const doLogin = useCallback(async (email: string, password: string) => {
    const data = await apiLogin(email, password)
    setToken(data.access_token)
    setTokenState(data.access_token)
    return data
  }, [])

  const doRegister = useCallback(async (email: string, password: string) => {
    const data = await apiRegister(email, password)
    setToken(data.access_token)
    setTokenState(data.access_token)
    return data
  }, [])

  const doLogout = useCallback(() => {
    clearToken()
    setTokenState(null)
  }, [])

  return { token, isLoggedIn, login: doLogin, register: doRegister, logout: doLogout }
}
