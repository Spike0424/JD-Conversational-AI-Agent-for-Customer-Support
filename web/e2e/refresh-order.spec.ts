import { test, expect, type Route } from '@playwright/test'

test('AFTER REFRESH + click history session + send → user msg must be above AI reply', async ({ page }) => {
  // Stub auth
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: 'mock.jwt.token' }) })
  )
  // Stub shops
  await page.route('**/v1/shops', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ id: 1001, shop_name: '测试旗舰店' }]) })
  )
  // Stub products
  await page.route('**/v1/products**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ goods_id: 9001, goods_name: '智能保温杯', price: 199 }]) })
  )

  // sessions list — has 1 real session
  await page.route('**/v1/sessions', (route: Route) =>
    route.request().method() === 'GET'
      ? route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            sessions: [{
              session_id: 'hist-1',
              title: '智能保温杯',
              created_at: '2026-08-18T00:00:00Z',
              last_message_at: '2026-08-18T00:01:00Z',
              message_count: 2,
            }],
          }),
        })
      : route.fallback()
  )

  // session messages (history)
  await page.route('**/v1/sessions/hist-1/messages', (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        session_id: 'hist-1',
        messages: [
          { role: 'user', content: 'old question', timestamp: null },
          { role: 'assistant', content: 'old reply', timestamp: null },
        ],
      }),
    })
  )

  // SSE stream
  await page.route('**/v1/chat/stream', async (route: Route) => {
    const sse = 'data: ' + JSON.stringify({ delta: 'NEW REPLY' }) + '\n\ndata: [DONE]\n\n'
    await route.fulfill({ status: 200, headers: { 'Content-Type': 'text/event-stream' }, body: sse })
  })

  await page.goto('/')

  // Login (since this is the first request, login Modal may not be present if JWT in storage;
  // for simplicity force login via modal)
  await page.getByRole('button', { name: '登录' }).first().click()
  await page.locator('.ant-modal-content input[type="email"]').fill('t@e.com')
  await page.locator('.ant-modal-content input[type="password"]').fill('p12345678')
  await page.locator('.ant-modal-content button[type="submit"]').click()
  await expect(page.locator('.ant-modal-content')).toHaveCount(0, { timeout: 5000 })

  // Click the history session (模拟刷新后点击)
  await page.locator('.ant-layout-sider').getByText('智能保温杯').first().click()

  // Wait for history to load
  await expect(page.getByText('old question')).toBeVisible({ timeout: 5000 })
  await expect(page.getByText('old reply')).toBeVisible({ timeout: 5000 })

  // Now send a NEW message
  await page.getByPlaceholder(/输入您的问题/).fill('new hi')
  await page.getByPlaceholder(/输入您的问题/).press('Enter')

  // Wait for AI reply
  await expect(page.getByText('NEW REPLY')).toBeVisible({ timeout: 10_000 })

  // Verify order in DOM: new hi must come AFTER old reply but BEFORE NEW REPLY
  const order = await page.evaluate(() => {
    const body = document.body.innerText
    const oldQ = body.indexOf('old question')
    const oldR = body.indexOf('old reply')
    const newQ = body.indexOf('new hi')
    const newR = body.indexOf('NEW REPLY')
    return {
      oldQ, oldR, newQ, newR,
      oldOrder: oldQ >= 0 && oldR >= 0 && oldQ < oldR,
      newOrder: oldR >= 0 && newR >= 0 && newQ < newR,  // ← the bug if false
    }
  })
  console.log('order:', order)
  expect(order.newOrder).toBe(true)
})