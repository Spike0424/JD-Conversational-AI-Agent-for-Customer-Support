import { test, expect, type Route } from '@playwright/test'

const SHOP = '测试旗舰店'
const PRODUCT = '智能保温杯'

test('streaming AND refreshed history both render markdown (bold headings)', async ({ page }) => {
  // Login
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: 'mock.jwt.token' }) })
  )
  await page.route('**/v1/shops', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ id: 1001, shop_name: SHOP }]) })
  )
  await page.route('**/v1/products**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ goods_id: 9001, goods_name: PRODUCT, price: 199 }]) })
  )

  // SSE — emit model-style markdown chunk by chunk
  await page.route('**/v1/chat/stream', async (route: Route) => {
    const sse = [
      'data: ' + JSON.stringify({ delta: '**1. 性能强劲**\n' }) + '\n\n',
      'data: ' + JSON.stringify({ delta: '- 搭载 A19 芯片\n' }) + '\n\n',
      'data: ' + JSON.stringify({ delta: '- 配合 iOS 26\n' }) + '\n\n',
      'data: [DONE]\n\n',
    ].join('')
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' },
      body: sse,
    })
  })

  // sessions list — store nothing initially; after first send, add real session
  let savedSessions: { session_id: string; title: string; created_at: string; last_message_at: string; message_count: number }[] = []
  await page.route('**/v1/sessions', async (route: Route) => {
    if (route.request().method() === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ sessions: savedSessions }),
      })
    }
    return route.fallback()
  })

  // session messages — return parsed markdown that frontend should render
  await page.route('**/v1/sessions/sess-markdown/messages', (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        session_id: 'sess-markdown',
        messages: [
          { role: 'user', content: 'markdown test', timestamp: null },
          { role: 'assistant', content: '**1. 性能强劲**\n\n- 搭载 A19 芯片\n- 配合 iOS 26', timestamp: null },
        ],
      }),
    })
  )

  await page.goto('/')
  await page.getByRole('button', { name: '登录' }).first().click()
  await page.locator('.ant-modal-content input[type="email"]').fill('t@e.com')
  await page.locator('.ant-modal-content input[type="password"]').fill('p12345678')
  await page.locator('.ant-modal-content button[type="submit"]').click()
  await expect(page.locator('.ant-modal-content')).toHaveCount(0, { timeout: 5000 })

  // --- Phase 1: streaming ---
  await page.locator('.ant-select-selector').first().click()
  await page.locator('.ant-select-item-option-content', { hasText: SHOP }).click()
  await page.locator('.ant-select-selector').nth(1).click()
  await page.locator('.ant-select-item-option-content', { hasText: PRODUCT }).click()
  await page.getByRole('button', { name: '开始咨询' }).click()

  await page.getByPlaceholder(/输入您的问题/).fill('markdown test')
  await page.getByPlaceholder(/输入您的问题/).press('Enter')

  // Wait for streamed markdown to render <strong>
  await expect(page.locator('strong').filter({ hasText: '1. 性能强劲' }).first()).toBeVisible({ timeout: 10_000 })

  // Snapshot the rendered streaming HTML
  const streamHtml = await page.evaluate(() => {
    const assistantBubble = Array.from(document.querySelectorAll('div'))
      .find((d) => d.textContent?.includes('1. 性能强劲') && d.querySelector('strong'))
    return assistantBubble?.outerHTML || 'NOT FOUND'
  })
  console.log('=== Streaming HTML ===')
  console.log(streamHtml)
  // Verify streaming has <strong>
  expect(streamHtml).toContain('<strong')
  expect(streamHtml).toContain('1. 性能强劲')
  // Now rendered as paragraph not bullets
  expect(streamHtml).toContain('搭载 A19 芯片')

  // --- Phase 2: refresh + load history ---
  savedSessions = [{
    session_id: 'sess-markdown',
    title: PRODUCT,
    created_at: '2026-08-20T00:00:00Z',
    last_message_at: '2026-08-20T00:01:00Z',
    message_count: 2,
  }]
  await page.reload()
  // Token is in localStorage; login modal likely won't appear. Wait for chat area.
  await page.locator('.ant-layout-sider').getByText(PRODUCT).first().click()
  await expect(page.locator('strong').filter({ hasText: '1. 性能强劲' }).first()).toBeVisible({ timeout: 10_000 })

  const refreshHtml = await page.evaluate(() => {
    const assistantBubble = Array.from(document.querySelectorAll('div'))
      .find((d) => d.textContent?.includes('1. 性能强劲') && d.querySelector('strong'))
    return assistantBubble?.outerHTML || 'NOT FOUND'
  })
  console.log('=== Refreshed HTML ===')
  console.log(refreshHtml)
  // Debug: verify the content text directly
  const refreshContent = await page.evaluate(() => {
    const strong = document.querySelector('strong')
    const lis = document.querySelectorAll('li')
    return {
      strongText: strong?.textContent,
      liCount: lis.length,
      liTexts: Array.from(lis).map((li) => li.textContent),
      assistantBubbleHTML: Array.from(document.querySelectorAll('div'))
        .find((d) => d.textContent?.includes('1. 性能强劲'))?.innerHTML,
    }
  })
  console.log('=== Refresh content ===')
  console.log(JSON.stringify(refreshContent, null, 2))
  // Verify refresh also has <strong>
  expect(refreshHtml).toContain('<strong')
  expect(refreshHtml).toContain('1. 性能强劲')
  expect(refreshHtml).toContain('搭载 A19 芯片')
})