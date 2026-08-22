import { useCallback, useMemo, useState } from 'react'
import {
  clearToken,
  getToken,
  login as apiLogin,
  register as apiRegister,
  setToken,
} from '../service/api'

/**
 * Decode the `sub` claim from a JWT without verifying the signature.
 * The backend already validates the token on every request; we just
 * need the email to display it in the sidebar user popover.
 */
function decodeEmail(token: string | null): string | null {
  if (!token) return null
  const parts = token.split('.')
  if (parts.length < 2) return null
  try {
    const payload = JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/'))) as { sub?: string }
    return payload.sub ?? null
  } catch {
    return null
  }
}

export function useAuth() {
  const [token, setTokenState] = useState<string | null>(getToken())
  const isLoggedIn = !!token
  const email = useMemo(() => decodeEmail(token), [token])

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

  return { token, isLoggedIn, email, login: doLogin, register: doRegister, logout: doLogout }
}
