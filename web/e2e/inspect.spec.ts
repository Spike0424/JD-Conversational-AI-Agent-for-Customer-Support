import { test, type Route } from '@playwright/test'

test('inspect rendered HTML', async ({ page }) => {
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: 'mock.jwt.token' }) })
  )
  await page.route('**/v1/shops', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ id: 1001, shop_name: '测试旗舰店' }]) })
  )
  await page.route('**/v1/products**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ goods_id: 9001, goods_name: '智能保温杯', price: 199 }]) })
  )
  await page.route('**/v1/chat/stream', async (route: Route) => {
    const sse = 'data: ' + JSON.stringify({ delta: '**1. 性能与屏幕**：搭载苹果A19芯片。\n\n- 项目1\n- 项目2\n\n**2. 影像**：f/1.6。' }) + '\n\ndata: [DONE]\n\n'
    await route.fulfill({ status: 200, headers: { 'Content-Type': 'text/event-stream' }, body: sse })
  })
  await page.route('**/v1/sessions', (route: Route) =>
    route.request().method() === 'GET'
      ? route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) })
      : route.fallback()
  )

  await page.goto('/')
  await page.getByRole('button', { name: '登录' }).first().click()
  await page.locator('.ant-modal-content input[type="email"]').fill('t@e.com')
  await page.locator('.ant-modal-content input[type="password"]').fill('p12345678')
  await page.locator('.ant-modal-content button[type="submit"]').click()
  await page.waitForTimeout(500)
  await page.locator('.ant-select-selector').first().click()
  await page.locator('.ant-select-item-option-content', { hasText: '测试旗舰店' }).click()
  await page.locator('.ant-select-selector').nth(1).click()
  await page.locator('.ant-select-item-option-content', { hasText: '智能保温杯' }).click()
  await page.getByRole('button', { name: '开始咨询' }).click()
  await page.getByPlaceholder(/输入您的问题/).fill('test')
  await page.getByPlaceholder(/输入您的问题/).press('Enter')
  await page.waitForTimeout(2000)

  // Get the assistant bubble's FULL HTML (no truncation)
  const html = await page.evaluate(() => {
    const all = Array.from(document.querySelectorAll('strong'))
    return all.map((s) => s.outerHTML).join('\n---\n')
  })
  console.log('=== STRONG outerHTML ===')
  console.log(html)

  // Also get the parent <p> wrapping
  const parentHtml = await page.evaluate(() => {
    const all = Array.from(document.querySelectorAll('strong'))
    return all.map((s) => s.parentElement?.outerHTML).join('\n---\n')
  })
  console.log('=== PARENT outerHTML ===')
  console.log(parentHtml)
})
